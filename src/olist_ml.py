"""Funciones compartidas entre el notebook y la interfaz (Streamlit).

Todo lo que transforma datos crudos en variables del modelo vive aquí, para que
el entrenamiento y el despliegue usen exactamente la misma lógica.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
DATES = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

# Perfil del vendedor: se calcula solo con pedidos cuyo resultado ya se conocía
# en el momento de la predicción. PRIOR_* son los pseudo-conteos del suavizado.
SELLER_PROFILE = ["seller_log_orders", "seller_late_rate", "seller_bad_rate", "seller_handling_mean",
                  "seller_log_price_mean", "seller_freight_ratio_mean", "seller_log_distance_mean",
                  "seller_same_state_share"]
PRIOR_N = 20
MIN_ORDERS_SEGMENT = 20
DIST_FILL_KM = 430  # distancia mediana (EDA 5.5) para los pocos pedidos sin coordenadas
NEW_SELLER_SEGMENT = "Nuevo / poco historial"

NUMERIC = ["price", "freight", "freight_ratio", "weight_g", "volume_cm3", "n_items", "n_sellers", "photos",
           "description_len", "installments", "distance_km", "promised_days", "approval_hours",
           "handling_days", "slack_at_ship", "limit_margin_days", "shipped_after_limit", "same_state",
           "ship_dow", "net_late_rate_28d", "net_backlog_ratio"] + SELLER_PROFILE
CATEGORICAL = ["customer_state", "seller_state", "category", "payment_type"]
SKEWED = ["price", "freight", "weight_g", "volume_cm3", "description_len", "approval_hours"]

# Nombres en lenguaje de negocio (gráficas SHAP e interfaz)
LABELS = {
    "price": "Valor de los productos", "freight": "Valor del flete", "freight_ratio": "Flete / valor",
    "weight_g": "Peso total", "volume_cm3": "Volumen total", "n_items": "Número de productos",
    "n_sellers": "Número de vendedores", "photos": "Fotos del producto", "description_len": "Largo de la descripción",
    "installments": "Cuotas de pago", "distance_km": "Distancia vendedor–cliente", "promised_days": "Días prometidos",
    "approval_hours": "Horas hasta aprobar el pago", "handling_days": "Días de alistamiento del vendedor",
    "slack_at_ship": "Días restantes hasta la fecha prometida", "limit_margin_days": "Margen frente al plazo de despacho",
    "shipped_after_limit": "Despachó después del plazo", "same_state": "Mismo estado vendedor y cliente",
    "ship_dow": "Día de la semana del despacho", "net_late_rate_28d": "% de retrasos de la red (28 días)",
    "net_backlog_ratio": "Congestión de la red", "seller_log_orders": "Experiencia del vendedor (pedidos)",
    "seller_late_rate": "Tasa histórica de retraso del vendedor", "seller_bad_rate": "Tasa histórica de reseñas malas",
    "seller_handling_mean": "Alistamiento promedio del vendedor", "seller_log_price_mean": "Precio típico del vendedor",
    "seller_freight_ratio_mean": "Flete típico del vendedor", "seller_log_distance_mean": "Distancia típica del vendedor",
    "seller_same_state_share": "% de ventas locales del vendedor", "customer_state": "Estado del cliente",
    "seller_state": "Estado del vendedor", "category": "Categoría", "payment_type": "Medio de pago",
    "seller_segment": "Segmento del vendedor",
}


def pretty_feature(name: str) -> str:
    """'log__price' → 'Valor de los productos'; 'cat__customer_state_RJ' → 'Estado del cliente = RJ'."""
    block, _, feat = name.partition("__")
    if block == "cat":
        for col in sorted(LABELS, key=len, reverse=True):
            if feat.startswith(col + "_"):
                return f"{LABELS[col]} = {feat[len(col) + 1:]}"
    return LABELS.get(feat, feat)


def download_kaggle() -> Path:
    """Descarga (o reutiliza de la caché) el dataset oficial de Kaggle y devuelve la carpeta."""
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def load_raw(folder: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(folder / f, parse_dates=DATES.get(name)) for name, f in FILES.items()}


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def zip_coordinates(geo: pd.DataFrame) -> pd.DataFrame:
    """Una coordenada mediana por prefijo postal, descartando puntos fuera de Brasil."""
    inside = geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)
    return (geo[inside]
            .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
            .median()
            .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"}))


def build_orders(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabla a nivel pedido entregado, con variables de compra, despacho y resultado."""
    orders = raw["orders"]
    translation = raw["category_translation"]
    en = dict(zip(translation.product_category_name, translation.product_category_name_english))
    products = raw["products"].assign(
        category=lambda p: p.product_category_name.map(en).fillna(p.product_category_name).fillna("unknown"))

    items = (raw["items"]
             .merge(products, on="product_id", how="left")
             .merge(raw["sellers"], on="seller_id", how="left")
             .sort_values(["order_id", "order_item_id"]))
    items["volume_cm3"] = items.product_length_cm * items.product_height_cm * items.product_width_cm

    items_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        price=("price", "sum"),
        freight=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"),
        volume_cm3=("volume_cm3", "sum"),
        photos=("product_photos_qty", "mean"),
        description_len=("product_description_lenght", "mean"),
        category=("category", "first"),
        seller_id=("seller_id", "first"),
        seller_state=("seller_state", "first"),
        seller_zip=("seller_zip_code_prefix", "first"),
        shipping_limit=("shipping_limit_date", "max"),
    )
    payments_agg = raw["payments"].groupby("order_id").agg(
        payment_type=("payment_type", lambda s: s.mode().iat[0]),
        installments=("payment_installments", "max"),
        n_payments=("payment_sequential", "max"),
    )
    reviews_last = (raw["reviews"]
                    .sort_values("review_answer_timestamp")
                    .drop_duplicates("order_id", keep="last")
                    .set_index("order_id")[["review_score", "review_creation_date", "review_answer_timestamp",
                                            "review_comment_message"]])
    coords = zip_coordinates(raw["geolocation"])

    valid = (
        (orders.order_status == "delivered")
        & orders.order_delivered_customer_date.notna()
        & orders.order_delivered_carrier_date.notna()
        & ~(orders.order_delivered_customer_date < orders.order_delivered_carrier_date)
    )
    df = (orders[valid]
          .merge(raw["customers"], on="customer_id")
          .join(items_agg, on="order_id", how="inner")
          .join(payments_agg, on="order_id")
          .join(reviews_last, on="order_id")
          .join(coords.add_prefix("customer_"), on="customer_zip_code_prefix")
          .join(coords.add_prefix("seller_"), on="seller_zip"))

    days = lambda a, b: (a - b).dt.total_seconds() / 86400
    df = df.assign(
        late=(df.order_delivered_customer_date.dt.normalize() > df.order_estimated_delivery_date).astype(int),
        bad_review=np.where(df.review_score.notna(), (df.review_score <= 2).astype(float), np.nan),
        purchase_month=df.order_purchase_timestamp.dt.to_period("M"),
        distance_km=haversine_km(df.customer_lat, df.customer_lng, df.seller_lat, df.seller_lng),
        same_state=(df.customer_state == df.seller_state).astype(int),
        freight_ratio=df.freight / df.price,
        promised_days=days(df.order_estimated_delivery_date, df.order_purchase_timestamp),
        approval_hours=days(df.order_approved_at, df.order_purchase_timestamp) * 24,
        handling_days=days(df.order_delivered_carrier_date, df.order_approved_at),
        days_to_ship=days(df.order_delivered_carrier_date, df.order_purchase_timestamp),
        slack_at_ship=days(df.order_estimated_delivery_date, df.order_delivered_carrier_date),
        limit_margin_days=days(df.shipping_limit, df.order_delivered_carrier_date),
        shipped_after_limit=(df.order_delivered_carrier_date > df.shipping_limit).astype(int),
        ship_dow=df.order_delivered_carrier_date.dt.dayofweek,
        delivery_days=days(df.order_delivered_customer_date, df.order_purchase_timestamp),
        delay_days=days(df.order_delivered_customer_date, df.order_estimated_delivery_date),
    )
    return df.sort_values("order_delivered_carrier_date").reset_index(drop=True)


def _asof_cumulative(df, events, time_col, value_cols, prefix):
    """Para cada pedido, suma acumulada de `value_cols` de los eventos del mismo vendedor
    ocurridos estrictamente antes de su despacho."""
    ev = events.sort_values(time_col)
    cum = ev.groupby("seller_id")[value_cols].cumsum().add_prefix(prefix)
    cum[prefix + "n"] = ev.groupby("seller_id").cumcount() + 1
    cum["seller_id"], cum["_t"] = ev["seller_id"].values, ev[time_col].values
    left = df[["order_id", "seller_id", "order_delivered_carrier_date"]].sort_values("order_delivered_carrier_date")
    out = pd.merge_asof(left, cum, left_on="order_delivered_carrier_date", right_on="_t", by="seller_id",
                        allow_exact_matches=False)
    return out.set_index("order_id").drop(columns=["seller_id", "order_delivered_carrier_date", "_t"])


def seller_history(df: pd.DataFrame, prior_late: float, prior_bad: float) -> pd.DataFrame:
    """Perfil de cada vendedor *en el momento del despacho* de cada pedido (sin mirar el futuro).

    - Retrasos: solo pedidos del vendedor ya entregados antes del despacho actual.
    - Reseñas: solo reseñas respondidas antes del despacho actual.
    - Operación (alistamiento, precio, flete, distancia): pedidos ya despachados antes.
    Las tasas se suavizan hacia el promedio global (`prior_*`) con PRIOR_N pseudo-pedidos,
    para que un vendedor con 1 pedido tarde no quede con tasa 100 %.
    """
    base = df.assign(log_price=np.log1p(df.price), log_dist=np.log1p(df.distance_km.fillna(DIST_FILL_KM)),
                     bad=df.bad_review)
    ops = _asof_cumulative(df, base.dropna(subset=["handling_days"]), "order_delivered_carrier_date",
                           ["handling_days", "log_price", "freight_ratio", "log_dist", "same_state"], "ops_")
    lat = _asof_cumulative(df, base, "order_delivered_customer_date", ["late"], "lat_")
    rev = _asof_cumulative(df, base.dropna(subset=["bad", "review_answer_timestamp"]),
                           "review_answer_timestamp", ["bad"], "rev_")
    h = ops.join(lat).join(rev).fillna(0)
    n_ops = h["ops_n"].replace(0, np.nan)
    prof = pd.DataFrame(index=h.index)
    prof["seller_known_orders"] = h["lat_n"]
    prof["seller_log_orders"] = np.log1p(h["lat_n"])
    prof["seller_late_rate"] = (h["lat_late"] + PRIOR_N * prior_late) / (h["lat_n"] + PRIOR_N)
    prof["seller_bad_rate"] = (h["rev_bad"] + PRIOR_N * prior_bad) / (h["rev_n"] + PRIOR_N)
    prof["seller_handling_mean"] = h["ops_handling_days"] / n_ops
    prof["seller_log_price_mean"] = h["ops_log_price"] / n_ops
    prof["seller_freight_ratio_mean"] = h["ops_freight_ratio"] / n_ops
    prof["seller_log_distance_mean"] = h["ops_log_dist"] / n_ops
    prof["seller_same_state_share"] = h["ops_same_state"] / n_ops
    return prof


def seller_snapshot(df: pd.DataFrame, cutoff, prior_late: float, prior_bad: float) -> pd.DataFrame:
    """Perfil de cada vendedor con lo que se sabía en `cutoff` (misma fórmula que `seller_history`).
    Se usa para entrenar la segmentación y para alimentar la interfaz en producción."""
    cutoff = pd.Timestamp(cutoff)
    ops = df[(df.order_delivered_carrier_date < cutoff) & df.handling_days.notna()].assign(
        log_price=lambda d: np.log1p(d.price),
        log_dist=lambda d: np.log1p(d.distance_km.fillna(DIST_FILL_KM)))
    lat = df[df.order_delivered_customer_date < cutoff]
    rev = df[(df.review_answer_timestamp < cutoff) & df.bad_review.notna()]
    o = ops.groupby("seller_id").agg(
        seller_handling_mean=("handling_days", "mean"),
        seller_log_price_mean=("log_price", "mean"),
        seller_freight_ratio_mean=("freight_ratio", "mean"),
        seller_log_distance_mean=("log_dist", "mean"),
        seller_same_state_share=("same_state", "mean"),
        seller_state=("seller_state", "last"),
        seller_zip=("seller_zip", "last"),
    )
    lt = lat.groupby("seller_id")["late"].agg(["size", "sum"])
    bd = rev.groupby("seller_id")["bad_review"].agg(["size", "sum"])
    prof = o.join(lt).join(bd, rsuffix="_rev").fillna({"size": 0, "sum": 0, "size_rev": 0, "sum_rev": 0})
    prof["seller_known_orders"] = prof["size"]
    prof["seller_log_orders"] = np.log1p(prof["size"])
    prof["seller_late_rate"] = (prof["sum"] + PRIOR_N * prior_late) / (prof["size"] + PRIOR_N)
    prof["seller_bad_rate"] = (prof["sum_rev"] + PRIOR_N * prior_bad) / (prof["size_rev"] + PRIOR_N)
    return prof[["seller_state", "seller_zip", "seller_known_orders"] + SELLER_PROFILE]


class Winsorizer(TransformerMixin, BaseEstimator):
    """Recorta cada columna a sus percentiles [lower, upper] aprendidos en `fit`."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def assign_segment(profiles: pd.DataFrame, segmenter, names: dict) -> pd.Series:
    """Segmento de negocio para cada fila de perfil; vendedores con poco historial → 'nuevo'."""
    seg = pd.Series(NEW_SELLER_SEGMENT, index=profiles.index, dtype=object)
    ok = (profiles["seller_known_orders"] >= MIN_ORDERS_SEGMENT) & profiles[SELLER_PROFILE].notna().all(axis=1)
    if ok.any():
        seg[ok] = pd.Series(segmenter.predict(profiles.loc[ok, SELLER_PROFILE]), index=profiles.index[ok]).map(names)
    return seg


def network_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Congestión logística conocida en el momento del despacho de cada pedido.

    - net_late_rate_28d: % de entregas tardías entre los pedidos entregados en los 28 días previos.
    - net_backlog_ratio: pedidos en tránsito (despachados y aún no entregados) / despachos de los 7 días previos.
    """
    t = df["order_delivered_carrier_date"].values
    shipped = np.sort(df["order_delivered_carrier_date"].values)
    deliv_order = np.argsort(df["order_delivered_customer_date"].values)
    delivered = df["order_delivered_customer_date"].values[deliv_order]
    late_cum = np.concatenate([[0], np.cumsum(df["late"].values[deliv_order])])

    def count_before(arr, x):
        return np.searchsorted(arr, x, side="left")

    win = np.timedelta64(28, "D")
    d_hi, d_lo = count_before(delivered, t), count_before(delivered, t - win)
    n_win = d_hi - d_lo
    late_win = late_cum[d_hi] - late_cum[d_lo]
    in_transit = count_before(shipped, t) - d_hi
    shipped_7d = count_before(shipped, t) - count_before(shipped, t - np.timedelta64(7, "D"))
    return pd.DataFrame({
        "net_late_rate_28d": np.where(n_win > 0, late_win / np.maximum(n_win, 1), np.nan),
        "net_backlog_ratio": in_transit / np.maximum(shipped_7d, 1),
    }, index=df["order_id"].values)


def features_from_form(form: dict, seller: pd.Series | None, coords: pd.DataFrame, network: dict) -> pd.DataFrame:
    """Convierte los datos de un pedido (formulario de la interfaz) en la fila de variables del modelo.

    `form` usa los mismos nombres que la tabla de pedidos (fechas como Timestamp).
    `seller` es la fila del vendedor en la foto de perfiles (o `new_seller_profile` si no tiene historial).
    """
    f = dict(form)
    days = lambda a, b: (pd.Timestamp(a) - pd.Timestamp(b)).total_seconds() / 86400
    ship = pd.Timestamp(f["order_delivered_carrier_date"])
    to_zip = lambda z: int(z) if pd.notna(z) else -1
    c = coords.reindex([to_zip(f.get("customer_zip_code_prefix"))]).iloc[0]
    s = coords.reindex([to_zip(f.get("seller_zip"))]).iloc[0]
    row = {
        "price": f["price"], "freight": f["freight"], "freight_ratio": f["freight"] / max(f["price"], 0.01),
        "weight_g": f["weight_g"], "volume_cm3": f["length_cm"] * f["height_cm"] * f["width_cm"],
        "n_items": f["n_items"], "n_sellers": 1, "photos": f["photos"], "description_len": f["description_len"],
        "installments": f["installments"],
        "distance_km": float(haversine_km(c.lat, c.lng, s.lat, s.lng)),
        "promised_days": days(f["order_estimated_delivery_date"], f["order_purchase_timestamp"]),
        "approval_hours": days(f["order_approved_at"], f["order_purchase_timestamp"]) * 24,
        "handling_days": days(ship, f["order_approved_at"]),
        "slack_at_ship": days(f["order_estimated_delivery_date"], ship),
        "limit_margin_days": days(f["shipping_limit"], ship),
        "shipped_after_limit": int(ship > pd.Timestamp(f["shipping_limit"])),
        "same_state": int(f["customer_state"] == f["seller_state"]),
        "ship_dow": ship.dayofweek,
        "customer_state": f["customer_state"], "seller_state": f["seller_state"],
        "category": f["category"], "payment_type": f["payment_type"],
        **network,
    }
    for col in ["seller_known_orders"] + SELLER_PROFILE:
        row[col] = seller[col]
    return pd.DataFrame([row])


def new_seller_profile(prior_late: float, prior_bad: float) -> pd.Series:
    """Perfil de un vendedor sin historial: igual al que `seller_history` asigna en su primer pedido."""
    p = pd.Series(np.nan, index=["seller_known_orders"] + SELLER_PROFILE)
    p[["seller_known_orders", "seller_log_orders"]] = 0
    p[["seller_late_rate", "seller_bad_rate"]] = prior_late, prior_bad
    return p


def psi(expected, actual, bins=10):
    """Population Stability Index entre una muestra de referencia y una actual."""
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
