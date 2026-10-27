"""Interfaz para el equipo de operaciones: riesgo de retraso de un pedido al momento del despacho.

Ejecutar desde la carpeta del proyecto:  streamlit run app/app.py
"""
import io
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import olist_ml as ml  # noqa: E402

st.set_page_config(page_title="Alerta de retrasos · Olist", page_icon="🚚", layout="wide")

RISK_STYLE = {
    "alto": ("🔴 Riesgo ALTO", "#c0392b",
             "Contactar hoy al cliente para ajustar la expectativa y abrir seguimiento con la transportadora."),
    "medio": ("🟡 Riesgo MEDIO", "#d68910",
              "Vigilar: revisar el rastreo en 48 horas y escalar si el paquete no avanza."),
    "bajo": ("🟢 Riesgo BAJO", "#1e8449", "Sin acción especial."),
}
FORM_COLUMNS = ["seller_id", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                "order_estimated_delivery_date", "shipping_limit", "price", "freight", "weight_g", "length_cm",
                "height_cm", "width_cm", "n_items", "photos", "description_len", "installments", "customer_state",
                "customer_zip_code_prefix", "category", "payment_type"]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "late_delivery_model.joblib")


@st.cache_resource
def load_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


B = load_bundle()
MODEL = B["model"]
PREP, CLF = MODEL.named_steps["prep"], MODEL.named_steps["model"]
SELLERS = B["sellers"]
T_HIGH, T_MED = B["thresholds"]["alto"], B["thresholds"]["medio"]


def risk_level(p):
    return "alto" if p >= T_HIGH else "medio" if p >= T_MED else "bajo"


def original_column(transformed):
    block, _, feat = transformed.partition("__")
    if block != "cat":
        return feat
    return next(c for c in sorted(B["features"]["cat"], key=len, reverse=True) if feat.startswith(c + "_"))


def explain(rows: pd.DataFrame) -> pd.DataFrame:
    """Contribución SHAP de cada variable original (las columnas one-hot se suman)."""
    sv = load_explainer(CLF)(PREP.transform(rows)).values
    if sv.ndim == 3:
        sv = sv[..., 1]
    cols = [original_column(n) for n in PREP.get_feature_names_out()]
    return pd.DataFrame(sv, columns=cols).T.groupby(level=0).sum().T


def fmt_value(col, v):
    if pd.isna(v):
        return "sin dato"
    if col in ("shipped_after_limit", "same_state"):
        return "sí" if v else "no"
    if col == "ship_dow":
        return ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"][int(v)]
    if col in ("price", "freight"):
        return f"R$ {v:,.2f}"
    if col in ("seller_late_rate", "seller_bad_rate", "seller_same_state_share", "net_late_rate_28d", "freight_ratio"):
        return f"{v:.0%}"
    if col == "seller_log_orders":
        return f"{np.expm1(v):,.0f} pedidos"
    if isinstance(v, str):
        return v
    return f"{v:,.1f}"


def reasons_text(contrib: pd.Series, row: pd.Series, n_up=4, n_down=2):
    up = contrib[contrib > 0].sort_values(ascending=False).head(n_up)
    down = contrib[contrib < 0].sort_values().head(n_down)
    lines = [f"⬆️ **{ml.LABELS.get(c, c)}**: {fmt_value(c, row[c])}" for c in up.index]
    lines += [f"⬇️ **{ml.LABELS.get(c, c)}**: {fmt_value(c, row[c])}" for c in down.index]
    return lines


def seller_profile(seller_id):
    if seller_id in SELLERS.index:
        return SELLERS.loc[seller_id]
    return ml.new_seller_profile(B["priors"]["late"], B["priors"]["bad"])


def typical_seller_zip(state):
    """Prefijo postal más común de los vendedores del estado (para ubicar a un vendedor nuevo)."""
    z = SELLERS.loc[SELLERS.seller_state == state, "seller_zip"]
    return z.mode().iat[0] if len(z) else np.nan


def build_row(form):
    return ml.features_from_form(form, seller_profile(form.get("seller_id")), B["coords"], B["network"])


def show_result(row: pd.DataFrame, seller_id=None, actual=None):
    p = float(MODEL.predict_proba(row)[0, 1])
    level = risk_level(p)
    title, color, action = RISK_STYLE[level]
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown(f"<div style='padding:1rem;border-radius:.6rem;background:{color};color:white'>"
                    f"<div style='font-size:1.4rem;font-weight:700'>{title}</div>"
                    f"<div style='font-size:2.4rem;font-weight:800'>{p:.0%}</div>"
                    f"<div>probabilidad de llegar después de la fecha prometida</div></div>",
                    unsafe_allow_html=True)
        st.markdown(f"**Acción sugerida:** {action}")
        if actual is not None:
            st.info(f"Resultado real de este pedido: **{'llegó tarde' if actual else 'llegó a tiempo'}**")
        seg = SELLERS.loc[seller_id, "segment"] if seller_id in SELLERS.index else ml.NEW_SELLER_SEGMENT
        st.markdown(f"**Segmento del vendedor:** {seg}")
    with c2:
        contrib = explain(row).iloc[0]
        st.markdown("**¿Por qué?** Factores que más mueven el riesgo de este pedido:")
        for line in reasons_text(contrib, row.iloc[0]):
            st.markdown(line)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])
        table = pd.DataFrame({
            "Factor": [ml.LABELS.get(c, c) for c in top.index],
            "En este pedido": [fmt_value(c, row.iloc[0][c]) for c in top.index],
            "Efecto": ["aumenta el riesgo" if v > 0 else "reduce el riesgo" for v in top.values],
            "Peso": top.abs().values,
        })
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Peso": st.column_config.ProgressColumn("Peso en la decisión", min_value=0,
                                                    max_value=float(table.Peso.max()), format=" ")})
        st.caption("Los 8 factores que más pesan en esta predicción, calculados con SHAP.")


def seller_label(sid):
    s = SELLERS.loc[sid]
    return f"{sid[:8]}… · {s.seller_state} · {int(s.seller_known_orders)} pedidos · {s.segment}"


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🚚 Alerta de retrasos")
    st.write("Estima, en el momento en que el vendedor entrega el paquete a la transportadora, "
             "la probabilidad de que el pedido llegue **después de la fecha prometida al cliente**.")
    m = B["metrics"]
    st.metric("Retrasos detectados revisando el 10 % diario", f"{m['headline']['recall']:.0%}")
    st.metric("Aciertos entre los marcados", f"{m['headline']['precisión']:.0%}",
              help=f"Frente a {m['test_prevalence']:.1%} si se marcara al azar")
    st.caption(f"Evaluado en pedidos reales de may–ago 2018 (PR-AUC {m['test_pr_auc']:.2f}, "
               f"ROC-AUC {m['test_roc_auc']:.2f}). Modelo v{B['version']}, entrenado con datos hasta "
               f"{B['trained_until']:%d-%m-%Y}.")
    st.caption("**Riesgo alto:** puntaje dentro del 10 % más alto de los despachos recientes. "
               "**Medio:** dentro del 25 %. En el lote del día se marca directamente el 10 % de mayor riesgo.")

tab_one, tab_batch, tab_examples, tab_segments = st.tabs(
    ["Evaluar un despacho", "Lote del día (CSV)", "Ejemplos reales", "Segmentos de vendedores"])

# ------------------------------------------------------------------ un pedido
with tab_one:
    st.subheader("Datos del despacho")
    ids = SELLERS.sort_values("seller_known_orders", ascending=False).index.tolist()
    with st.form("pedido"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Vendedor**")
            new_seller = st.checkbox("Vendedor nuevo (sin historial)")
            sid = st.selectbox("Vendedor", ids, format_func=seller_label, disabled=new_seller)
            states = B["choices"]["customer_state"]
            seller_state = st.selectbox("Estado del vendedor (si es nuevo)", states, index=states.index("SP"),
                                        disabled=not new_seller)
            st.markdown("**Cliente**")
            customer_state = st.selectbox("Estado del cliente", states, index=states.index("RJ"))
            zip_default = int(B["choices"]["customer_zip_by_state"][customer_state])
            customer_zip = st.number_input("Prefijo postal del cliente (5 dígitos)", 1000, 99999, zip_default,
                                           help="Por defecto, el prefijo más común del estado elegido.")
        with c2:
            st.markdown("**Producto y pago**")
            cats = B["choices"]["category"]
            category = st.selectbox("Categoría", cats, index=cats.index("bed_bath_table"))
            price = st.number_input("Valor de los productos (R$)", 1.0, 20000.0, 120.0)
            freight = st.number_input("Valor del flete (R$)", 0.0, 2000.0, 20.0)
            weight = st.number_input("Peso total (g)", 0, 200000, 1200)
            d1, d2, d3 = st.columns(3)
            length = d1.number_input("Largo (cm)", 1, 200, 30)
            height = d2.number_input("Alto (cm)", 1, 200, 15)
            width = d3.number_input("Ancho (cm)", 1, 200, 20)
            n_items = st.number_input("Número de productos", 1, 20, 1)
            pays = B["choices"]["payment_type"]
            payment = st.selectbox("Medio de pago", pays, index=pays.index("credit_card"))
            installments = st.number_input("Cuotas", 1, 24, 3)
        with c3:
            st.markdown("**Fechas**")
            today = date(2018, 8, 20)
            purchase = st.date_input("Fecha de compra", today - timedelta(days=4))
            approved = st.date_input("Fecha de aprobación del pago", today - timedelta(days=4))
            ship = st.date_input("Fecha de entrega a la transportadora (hoy)", today)
            limit = st.date_input("Plazo máximo de despacho del vendedor", today - timedelta(days=1))
            promised = st.date_input("Fecha de entrega prometida al cliente", today + timedelta(days=12))
            with st.expander("Datos opcionales"):
                photos = st.number_input("Fotos del producto", 0, 20, 2)
                desc = st.number_input("Largo de la descripción (caracteres)", 0, 4000, 800)
        submitted = st.form_submit_button("Calcular riesgo", type="primary")

    if submitted:
        noon = time(12, 0)
        seller_id = None if new_seller else sid
        prof = seller_profile(seller_id)
        form = {
            "seller_id": seller_id,
            "order_purchase_timestamp": datetime.combine(purchase, time(10, 0)),
            "order_approved_at": datetime.combine(approved, time(11, 0)),
            "order_delivered_carrier_date": datetime.combine(ship, noon),
            "order_estimated_delivery_date": datetime.combine(promised, time(0, 0)),
            "shipping_limit": datetime.combine(limit, time(23, 59)),
            "price": price, "freight": freight, "weight_g": weight, "length_cm": length, "height_cm": height,
            "width_cm": width, "n_items": n_items, "photos": photos, "description_len": desc,
            "installments": installments, "customer_state": customer_state, "customer_zip_code_prefix": customer_zip,
            "seller_state": seller_state if new_seller else SELLERS.loc[sid, "seller_state"],
            "seller_zip": typical_seller_zip(seller_state) if new_seller else SELLERS.loc[sid, "seller_zip"],
            "category": category, "payment_type": payment,
        }
        errors = []
        if not (form["order_purchase_timestamp"] <= form["order_approved_at"] <= form["order_delivered_carrier_date"]):
            errors.append("Las fechas deben ir en orden: compra ≤ aprobación ≤ despacho.")
        if form["order_estimated_delivery_date"] <= form["order_purchase_timestamp"]:
            errors.append("La fecha prometida debe ser posterior a la compra.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            show_result(build_row(form), seller_id)

# ------------------------------------------------------------------ lote
with tab_batch:
    st.subheader("Priorizar los despachos del día")
    st.write("Suba un archivo CSV con un despacho por fila. Descargue la plantilla para ver el formato; "
             "deje `seller_id` vacío para vendedores nuevos.")
    ex = B["examples"].head(3)
    template = pd.DataFrame({
        "seller_id": ex.seller_id, "order_purchase_timestamp": ex.order_purchase_timestamp,
        "order_approved_at": ex.order_approved_at, "order_delivered_carrier_date": ex.order_delivered_carrier_date,
        "order_estimated_delivery_date": ex.order_estimated_delivery_date, "shipping_limit": ex.shipping_limit,
        "price": ex.price, "freight": ex.freight, "weight_g": ex.weight_g, "length_cm": 30, "height_cm": 15,
        "width_cm": 20, "n_items": ex.n_items, "photos": ex.photos, "description_len": ex.description_len,
        "installments": ex.installments, "customer_state": ex.customer_state,
        "customer_zip_code_prefix": ex.customer_zip_code_prefix, "category": ex.category,
        "payment_type": ex.payment_type})[FORM_COLUMNS]
    st.download_button("Descargar plantilla CSV", template.to_csv(index=False).encode("utf-8"),
                       "plantilla_despachos.csv", "text/csv")
    up = st.file_uploader("Archivo de despachos", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in FORM_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Faltan columnas: {', '.join(missing)}")
        else:
            for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                      "order_estimated_delivery_date", "shipping_limit"]:
                data[c] = pd.to_datetime(data[c])
            out = []
            for _, r in data.iterrows():
                form = r.to_dict()
                sid = form["seller_id"] if isinstance(form["seller_id"], str) else None
                form["seller_id"] = sid
                form["seller_state"] = SELLERS.loc[sid, "seller_state"] if sid in SELLERS.index else "SP"
                form["seller_zip"] = (SELLERS.loc[sid, "seller_zip"] if sid in SELLERS.index
                                      else typical_seller_zip(form["seller_state"]))
                out.append(build_row(form))
            rows = pd.concat(out, ignore_index=True)
            probs = MODEL.predict_proba(rows)[:, 1]
            contrib = explain(rows)
            n_top = max(1, int(np.ceil(B["thresholds"]["capacidad"] * len(probs))))
            result = data.assign(
                probabilidad_retraso=probs.round(3),
                revisar_hoy=pd.Series(probs).rank(ascending=False, method="first").le(n_top).map({True: "sí", False: ""}),
                riesgo=[risk_level(p) for p in probs],
                segmento_vendedor=[SELLERS.loc[s, "segment"] if s in SELLERS.index else ml.NEW_SELLER_SEGMENT
                                   for s in data.seller_id],
                razon_principal=[ml.LABELS.get(c, c) for c in contrib.idxmax(axis=1)],
            ).sort_values("probabilidad_retraso", ascending=False)
            st.caption(f"'revisar_hoy' marca los {n_top} despachos de mayor riesgo (10 % del lote, "
                       "la capacidad diaria del equipo).")
            st.dataframe(result[["revisar_hoy", "probabilidad_retraso", "riesgo", "razon_principal", "segmento_vendedor",
                                 "seller_id", "customer_state", "order_estimated_delivery_date"]],
                         width="stretch")
            st.download_button("Descargar lista priorizada", result.to_csv(index=False).encode("utf-8"),
                               "despachos_priorizados.csv", "text/csv")

# ------------------------------------------------------------------ ejemplos
with tab_examples:
    st.subheader("Pedidos reales del periodo de prueba (may–ago 2018)")
    st.write("El modelo nunca vio estos pedidos durante el entrenamiento.")
    exs = B["examples"]
    labels = [f"Pedido {i + 1} · riesgo {r.riesgo} · {'llegó tarde' if r.late else 'llegó a tiempo'} · "
              f"{r.customer_state}" for i, r in exs.iterrows()]
    i = st.selectbox("Elija un pedido", range(len(exs)), format_func=lambda k: labels[k])
    r = exs.iloc[[i]]
    o = r.iloc[0]
    st.markdown(f"Compra **{o.order_purchase_timestamp:%d-%m-%Y}** · despacho **{o.order_delivered_carrier_date:%d-%m-%Y}** · "
                f"prometido **{o.order_estimated_delivery_date:%d-%m-%Y}** · {o.category} · R$ {o.price:,.2f} · "
                f"de {o.seller_state} a {o.customer_state} ({o.distance_km:,.0f} km)")
    show_result(r[B["features"]["num"] + B["features"]["cat"]], o.seller_id, actual=bool(o.late))

# ------------------------------------------------------------------ segmentos
with tab_segments:
    st.subheader("Segmentos de vendedores (K-Means sobre su historial)")
    st.write("Los vendedores con al menos 20 pedidos entregados se agrupan según su comportamiento. "
             "Sirve para decidir a quién acompañar, además de explicar el riesgo de cada pedido.")
    prof = B["segment_profiles"].copy()
    nice = prof.rename(columns={
        "vendedores": "Vendedores", "pedidos_mediana": "Pedidos (mediana)", "tasa_retraso": "Tasa de retraso",
        "tasa_resena_mala": "Reseñas 1–2", "alistamiento_dias": "Alistamiento (días)",
        "precio_tipico": "Precio típico (R$)", "flete_sobre_precio": "Flete / precio",
        "distancia_km": "Distancia típica (km)", "ventas_locales": "Ventas en su estado"})
    st.dataframe(nice.style.format({"Tasa de retraso": "{:.1%}", "Reseñas 1–2": "{:.1%}", "Flete / precio": "{:.0%}",
                                    "Ventas en su estado": "{:.0%}", "Alistamiento (días)": "{:.1f}",
                                    "Precio típico (R$)": "{:,.0f}", "Distancia típica (km)": "{:,.0f}",
                                    "Pedidos (mediana)": "{:,.0f}"}), width="stretch")
    counts = SELLERS.segment.value_counts()
    st.bar_chart(counts, horizontal=True, height=250)
    st.caption(f"Vendedores en la foto actual ({B['data_until']:%d-%m-%Y}). "
               "'Nuevo / poco historial' = menos de 20 pedidos entregados.")
