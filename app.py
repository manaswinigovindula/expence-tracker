# =============================================================================
# EXPENSE TRACKER - a single-file Streamlit app (frontend + backend together)
# Run with:  streamlit run app.py
# =============================================================================

# =============================================================================
# 1. IMPORTS
# =============================================================================
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: safe for web apps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sqlalchemy import (
    Date,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# =============================================================================
# 2. PAGE CONFIGURATION  (must be the first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants used all over the app
CATEGORIES = [
    "Food", "Transportation", "Shopping", "Bills", "Entertainment",
    "Health", "Education", "Travel", "Other",
]
PAYMENT_METHODS = ["Cash", "Credit Card", "Debit Card", "UPI", "Bank Transfer", "Other"]
CURRENCIES = ["₹", "$", "€", "£", "¥"]
COLUMNS = ["id", "title", "amount", "category", "expense_date", "payment_method", "description"]
SAMPLE_TAG = "[SAMPLE]"  # Sample rows start with this text so they are easy to remove

# =============================================================================
# 3. CUSTOM CSS
# =============================================================================
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        padding: 1.2rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; color: #ffffff; }
    .main-header p  { margin: 0.25rem 0 0 0; color: #E0E7FF; }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 4. DATABASE CONFIGURATION
# =============================================================================
# The database file is stored next to app.py and is created automatically.
DB_PATH = Path(__file__).resolve().parent / "expenses.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"


@st.cache_resource
def get_engine():
    """Create the database engine once and reuse it on every Streamlit rerun."""
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def get_session() -> Session:
    """Return a new database session (use it with a 'with' block)."""
    return Session(get_engine(), expire_on_commit=False)


# =============================================================================
# 5. SQLALCHEMY MODELS  (each class = one table)
# =============================================================================
class Base(DeclarativeBase):
    pass


class Expense(Base):
    __tablename__ = "expenses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    expense_date: Mapped[date] = mapped_column(Date, nullable=False)
    payment_method: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")


class Setting(Base):
    """Simple key/value table used for the monthly budget and currency."""
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[str] = mapped_column(String(200), nullable=False, default="")


# =============================================================================
# 6. DATABASE FUNCTIONS
# =============================================================================
def initialize_database() -> bool:
    """Create the database file and tables if they do not exist yet."""
    try:
        Base.metadata.create_all(get_engine())
        return True
    except SQLAlchemyError as exc:
        st.error(f"Could not initialize the database: {exc}")
        return False


def validate_expense(title, amount, category, expense_date, payment_method) -> list:
    """Check the inputs and return a list of error messages (empty = valid)."""
    errors = []
    if not title or not str(title).strip():
        errors.append("Expense title cannot be empty.")
    elif len(str(title).strip()) > 100:
        errors.append("Expense title must be 100 characters or fewer.")
    if amount is None or not np.isfinite(amount) or float(amount) <= 0:
        errors.append("Amount must be greater than 0.")
    if category not in CATEGORIES:
        errors.append("Please choose a valid category.")
    if payment_method not in PAYMENT_METHODS:
        errors.append("Please choose a valid payment method.")
    if not isinstance(expense_date, date):
        errors.append("Please choose a valid date.")
    return errors


def add_expense(title, amount, category, expense_date, payment_method, description):
    """Insert a new expense. Returns (success: bool, message: str)."""
    errors = validate_expense(title, amount, category, expense_date, payment_method)
    if errors:
        return False, " ".join(errors)
    try:
        with get_session() as session:
            session.add(
                Expense(
                    title=title.strip(),
                    amount=round(float(amount), 2),
                    category=category,
                    expense_date=expense_date,
                    payment_method=payment_method,
                    description=(description or "").strip(),
                )
            )
            session.commit()
        return True, "Expense added successfully."
    except SQLAlchemyError as exc:
        return False, f"Database error: {exc}"


def get_expenses() -> pd.DataFrame:
    """Read all expenses from the database into a pandas DataFrame."""
    data = []
    try:
        with get_session() as session:
            rows = session.scalars(
                select(Expense).order_by(Expense.expense_date.desc(), Expense.id.desc())
            ).all()
            data = [
                {
                    "id": r.id,
                    "title": r.title,
                    "amount": r.amount,
                    "category": r.category,
                    "expense_date": r.expense_date,
                    "payment_method": r.payment_method,
                    "description": r.description or "",
                }
                for r in rows
            ]
    except SQLAlchemyError as exc:
        st.error(f"Could not load expenses: {exc}")

    df = pd.DataFrame(data, columns=COLUMNS)
    df["id"] = df["id"].astype(int)
    df["amount"] = df["amount"].astype(float)
    df["expense_date"] = pd.to_datetime(df["expense_date"])
    return df


def update_expense(expense_id, title, amount, category, expense_date, payment_method, description):
    """Update an existing expense. Returns (success, message)."""
    errors = validate_expense(title, amount, category, expense_date, payment_method)
    if errors:
        return False, " ".join(errors)
    try:
        with get_session() as session:
            expense = session.get(Expense, int(expense_id))
            if expense is None:
                return False, "That expense no longer exists."
            expense.title = title.strip()
            expense.amount = round(float(amount), 2)
            expense.category = category
            expense.expense_date = expense_date
            expense.payment_method = payment_method
            expense.description = (description or "").strip()
            session.commit()
        return True, "Expense updated successfully."
    except SQLAlchemyError as exc:
        return False, f"Database error: {exc}"


def delete_expense(expense_id):
    """Delete one expense. Returns (success, message)."""
    try:
        with get_session() as session:
            expense = session.get(Expense, int(expense_id))
            if expense is None:
                return False, "That expense no longer exists."
            session.delete(expense)
            session.commit()
        return True, "Expense deleted."
    except SQLAlchemyError as exc:
        return False, f"Database error: {exc}"


def delete_all_expenses():
    """Remove every expense (used by the Settings page)."""
    try:
        with get_session() as session:
            result = session.execute(delete(Expense))
            session.commit()
        return True, f"Deleted {result.rowcount} expenses."
    except SQLAlchemyError as exc:
        return False, f"Database error: {exc}"


def count_expenses() -> int:
    try:
        with get_session() as session:
            return session.scalar(select(func.count(Expense.id))) or 0
    except SQLAlchemyError:
        return 0


def get_setting(key: str, default: str = "") -> str:
    try:
        with get_session() as session:
            row = session.get(Setting, key)
            return row.value if row else default
    except SQLAlchemyError:
        return default


def set_setting(key: str, value) -> bool:
    try:
        with get_session() as session:
            row = session.get(Setting, key)
            if row:
                row.value = str(value)
            else:
                session.add(Setting(key=key, value=str(value)))
            session.commit()
        return True
    except SQLAlchemyError as exc:
        st.error(f"Could not save setting: {exc}")
        return False


def get_budget() -> float:
    try:
        return max(float(get_setting("monthly_budget", "0") or 0), 0.0)
    except ValueError:
        return 0.0


# ---- Sample data (clearly labeled, easy to remove from the Settings page) ----
# Each item: (days ago, title, amount, category, payment method, description)
SAMPLE_EXPENSES = [
    (1, "Groceries", 1250.00, "Food", "UPI", "Weekly groceries"),
    (2, "Bus pass", 450.00, "Transportation", "Cash", "Monthly bus pass"),
    (4, "Electricity bill", 1800.00, "Bills", "Bank Transfer", "Monthly bill"),
    (6, "Movie night", 600.00, "Entertainment", "Credit Card", "Movie and snacks"),
    (9, "New shoes", 2999.00, "Shopping", "Debit Card", "Running shoes"),
    (12, "Pharmacy", 380.00, "Health", "Cash", "Medicines"),
    (15, "Online course", 1499.00, "Education", "Credit Card", "Python course"),
    (20, "Lunch with friends", 750.00, "Food", "UPI", "Restaurant"),
    (28, "Train tickets", 1600.00, "Travel", "UPI", "Weekend trip"),
    (35, "Internet bill", 899.00, "Bills", "Bank Transfer", "Monthly bill"),
    (45, "Birthday gift", 1200.00, "Shopping", "Cash", "Gift for a friend"),
    (60, "Dinner out", 950.00, "Food", "Credit Card", "Family dinner"),
]


def add_sample_data():
    try:
        today = date.today()
        with get_session() as session:
            for days_ago, title, amount, category, method, desc in SAMPLE_EXPENSES:
                session.add(
                    Expense(
                        title=title,
                        amount=amount,
                        category=category,
                        expense_date=today - timedelta(days=days_ago),
                        payment_method=method,
                        description=f"{SAMPLE_TAG} {desc}",
                    )
                )
            session.commit()
        return True, f"Added {len(SAMPLE_EXPENSES)} sample expenses."
    except SQLAlchemyError as exc:
        return False, f"Database error: {exc}"


def remove_sample_data():
    try:
        with get_session() as session:
            result = session.execute(
                delete(Expense).where(Expense.description.like(f"{SAMPLE_TAG}%"))
            )
            session.commit()
        return True, f"Removed {result.rowcount} sample expenses."
    except SQLAlchemyError as exc:
        return False, f"Database error: {exc}"


# =============================================================================
# 7. UTILITY FUNCTIONS
# =============================================================================
def fmt_money(value: float) -> str:
    """Format a number with the chosen currency symbol, e.g. ₹1,250.00"""
    return f"{st.session_state.get('currency', '₹')}{value:,.2f}"


def set_flash(kind: str, message: str):
    """Store a message that will be shown after the next rerun."""
    st.session_state["flash"] = (kind, message)


def show_flash():
    """Show (and clear) the stored message, if any."""
    flash = st.session_state.pop("flash", None)
    if flash:
        kind, message = flash
        if kind == "success":
            st.success(message, icon="✅")
        else:
            st.error(message, icon="🚫")


def page_header(title: str, subtitle: str):
    st.markdown(
        f'<div class="main-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_table(df: pd.DataFrame, **kwargs):
    """Show a table full-width. Works on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def display_version(df: pd.DataFrame) -> pd.DataFrame:
    """Make a copy of the expense table with friendly column names for display."""
    out = df.copy()
    out["expense_date"] = out["expense_date"].dt.strftime("%Y-%m-%d")
    out = out.rename(
        columns={
            "id": "ID", "title": "Title", "amount": "Amount", "category": "Category",
            "expense_date": "Date", "payment_method": "Payment Method",
            "description": "Description",
        }
    )
    return out


def money_column_config() -> dict:
    symbol = st.session_state.get("currency", "₹")
    return {"Amount": st.column_config.NumberColumn("Amount", format=f"{symbol}%.2f")}


def get_current_month_total(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    today = date.today()
    mask = (df["expense_date"].dt.year == today.year) & (df["expense_date"].dt.month == today.month)
    return float(df.loc[mask, "amount"].sum())


def calculate_totals(df: pd.DataFrame) -> dict:
    """Return the main numbers shown on the dashboard."""
    if df.empty:
        return {"total": 0.0, "count": 0, "average": 0.0, "highest": 0.0, "month": 0.0}
    return {
        "total": float(df["amount"].sum()),
        "count": int(len(df)),
        "average": float(df["amount"].mean()),
        "highest": float(df["amount"].max()),
        "month": get_current_month_total(df),
    }


def get_summary_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Total and count of expenses grouped by a column (category, payment_method...)."""
    if df.empty:
        return pd.DataFrame(columns=[column, "total", "count"])
    out = df.groupby(column, as_index=False).agg(total=("amount", "sum"), count=("id", "count"))
    return out.sort_values("total", ascending=False).reset_index(drop=True)


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return get_summary_by(df, "category")


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "total", "count"])
    temp = df.assign(month=df["expense_date"].dt.strftime("%Y-%m"))
    out = temp.groupby("month", as_index=False).agg(total=("amount", "sum"), count=("id", "count"))
    return out.sort_values("month").reset_index(drop=True)


def budget_status(spent: float, budget: float):
    """Show a progress bar and a warning/success message for the budget."""
    if budget <= 0:
        st.info("No monthly budget set yet. Go to the 💰 Budget page to set one.")
        return
    pct = spent / budget * 100
    st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of budget used")
    if pct >= 100:
        st.error(f"🚨 You are over budget by {fmt_money(spent - budget)}!")
    elif pct >= 80:
        st.warning(f"⚠️ You have used {pct:.0f}% of your budget. Spend carefully!")
    else:
        st.success(f"✅ You are within budget. {fmt_money(budget - spent)} left.")


# ---- Chart helpers ----
def new_chart(width=8, height=4.2):
    sns.set_theme(style="whitegrid")
    return plt.subplots(figsize=(width, height))


def finish_chart(fig, title: str, ax):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    sns.despine(ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # free memory


def label_horizontal_bars(ax, values):
    symbol = st.session_state.get("currency", "₹")
    for i, v in enumerate(values):
        ax.text(v, i, f" {symbol}{v:,.0f}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.2)


# =============================================================================
# 8. DASHBOARD
# =============================================================================
def page_dashboard():
    page_header("🏠 Dashboard", f"Your spending overview - {date.today():%B %Y}")
    df = get_expenses()
    totals = calculate_totals(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("💵 Total Expenses", fmt_money(totals["total"]))
    c2.metric("🧾 Transactions", totals["count"])
    c3.metric("📈 Average Expense", fmt_money(totals["average"]))

    c4, c5 = st.columns(2)
    c4.metric("🔝 Highest Expense", fmt_money(totals["highest"]))
    c5.metric("📅 This Month", fmt_money(totals["month"]))

    st.subheader("Monthly Budget")
    budget_status(totals["month"], get_budget())

    st.subheader("Recent Transactions")
    if df.empty:
        st.info("No expenses yet. Add your first one from the ➕ Add Expense page "
                "(or load sample data from ⚙️ Settings).")
    else:
        recent = df.sort_values(["expense_date", "id"], ascending=False).head(5)
        show_table(display_version(recent), column_config=money_column_config())


# =============================================================================
# 9. ADD EXPENSE
# =============================================================================
def page_add_expense():
    page_header("➕ Add Expense", "Record a new expense")
    show_flash()

    # The form version number lets us reset (clear) the form after saving.
    v = st.session_state.get("add_form_version", 0)

    with st.form(f"add_expense_form_{v}"):
        left, right = st.columns(2)
        with left:
            title = st.text_input("Expense title *", max_chars=100,
                                  placeholder="e.g. Grocery shopping", key=f"add_title_{v}")
            amount = st.number_input(
                f"Amount ({st.session_state.get('currency', '₹')}) *",
                min_value=0.0, value=0.0, step=1.0, format="%.2f", key=f"add_amount_{v}",
            )
            category = st.selectbox("Category *", CATEGORIES, key=f"add_cat_{v}")
        with right:
            expense_date = st.date_input("Date *", value=date.today(), key=f"add_date_{v}")
            payment_method = st.selectbox("Payment method *", PAYMENT_METHODS, key=f"add_pay_{v}")
        description = st.text_area("Description (optional)", max_chars=500, key=f"add_desc_{v}")
        submitted = st.form_submit_button("💾 Save Expense", type="primary")

    if submitted:
        errors = validate_expense(title, amount, category, expense_date, payment_method)
        if errors:
            for err in errors:
                st.error(err, icon="🚫")
        else:
            ok, message = add_expense(title, amount, category, expense_date, payment_method, description)
            if ok:
                st.session_state["add_form_version"] = v + 1  # fresh, empty form
                set_flash("success", f"Saved '{title.strip()}' - {fmt_money(amount)}")
                st.rerun()
            else:
                st.error(message)


# =============================================================================
# 10. EXPENSE MANAGEMENT (list, search/filter/sort, edit, delete, export)
# =============================================================================
def expense_labels(df: pd.DataFrame) -> dict:
    """Map expense id -> readable label for select boxes."""
    labels = {}
    for _, r in df.sort_values("id", ascending=False).iterrows():
        labels[int(r["id"])] = (
            f"#{int(r['id'])} • {r['title']} • {fmt_money(r['amount'])} • "
            f"{r['expense_date']:%Y-%m-%d}"
        )
    return labels


def render_list_tab(df: pd.DataFrame):
    if df.empty:
        st.info("No expenses to show yet.")
        return

    with st.expander("🔎 Search, filter and sort", expanded=True):
        search = st.text_input("Search (title or description)", placeholder="Type to search...")
        f1, f2, f3 = st.columns(3)
        cats = f1.multiselect("Category", CATEGORIES)
        methods = f2.multiselect("Payment method", PAYMENT_METHODS)
        period = f3.selectbox(
            "Date", ["All time", "This month", "Last 30 days", "This year", "Custom range"]
        )

        start_date = end_date = None
        if period == "Custom range":
            min_d = df["expense_date"].min().date()
            max_d = max(df["expense_date"].max().date(), date.today())
            picked = st.date_input("Choose start and end date", value=(min_d, max_d))
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start_date, end_date = picked
            else:
                st.info("Pick an end date to apply the range.")

        s1, s2 = st.columns(2)
        sort_options = {"Date": "expense_date", "Amount": "amount", "Title": "title",
                        "Category": "category", "ID": "id"}
        sort_label = s1.selectbox("Sort by", list(sort_options.keys()))
        sort_order = s2.radio("Order", ["Descending", "Ascending"], horizontal=True)

    # ---- apply filters ----
    result = df.copy()
    if search.strip():
        q = search.strip()
        mask = (
            result["title"].str.contains(q, case=False, na=False, regex=False)
            | result["description"].str.contains(q, case=False, na=False, regex=False)
        )
        result = result[mask]
    if cats:
        result = result[result["category"].isin(cats)]
    if methods:
        result = result[result["payment_method"].isin(methods)]

    today = date.today()
    if period == "This month":
        result = result[(result["expense_date"].dt.year == today.year)
                        & (result["expense_date"].dt.month == today.month)]
    elif period == "Last 30 days":
        result = result[result["expense_date"] >= pd.Timestamp(today - timedelta(days=30))]
    elif period == "This year":
        result = result[result["expense_date"].dt.year == today.year]
    elif period == "Custom range" and start_date and end_date:
        result = result[(result["expense_date"] >= pd.Timestamp(start_date))
                        & (result["expense_date"] <= pd.Timestamp(end_date))]

    result = result.sort_values(sort_options[sort_label], ascending=(sort_order == "Ascending"))

    st.caption(f"Showing {len(result)} of {len(df)} expenses - "
               f"total {fmt_money(result['amount'].sum() if not result.empty else 0)}")
    if result.empty:
        st.warning("No expenses match your filters.")
    else:
        show_table(display_version(result), column_config=money_column_config())
        csv_data = display_version(result).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download these results as CSV", csv_data,
                           file_name="expenses_filtered.csv", mime="text/csv")


def render_edit_tab(df: pd.DataFrame):
    if df.empty:
        st.info("There are no expenses to edit.")
        return
    labels = expense_labels(df)
    selected_id = st.selectbox("Select an expense to edit", list(labels.keys()),
                               format_func=lambda i: labels[i])
    row = df[df["id"] == selected_id].iloc[0]
    cat_index = CATEGORIES.index(row["category"]) if row["category"] in CATEGORIES else len(CATEGORIES) - 1
    pay_index = (PAYMENT_METHODS.index(row["payment_method"])
                 if row["payment_method"] in PAYMENT_METHODS else len(PAYMENT_METHODS) - 1)

    with st.form(f"edit_form_{selected_id}"):
        left, right = st.columns(2)
        with left:
            title = st.text_input("Title *", value=row["title"], max_chars=100,
                                  key=f"edit_title_{selected_id}")
            amount = st.number_input("Amount *", min_value=0.0, value=float(row["amount"]),
                                     step=1.0, format="%.2f", key=f"edit_amount_{selected_id}")
            category = st.selectbox("Category *", CATEGORIES, index=cat_index,
                                    key=f"edit_cat_{selected_id}")
        with right:
            expense_date = st.date_input("Date *", value=row["expense_date"].date(),
                                         key=f"edit_date_{selected_id}")
            payment_method = st.selectbox("Payment method *", PAYMENT_METHODS, index=pay_index,
                                          key=f"edit_pay_{selected_id}")
        description = st.text_area("Description", value=row["description"], max_chars=500,
                                   key=f"edit_desc_{selected_id}")
        submitted = st.form_submit_button("💾 Save Changes", type="primary")

    if submitted:
        errors = validate_expense(title, amount, category, expense_date, payment_method)
        if errors:
            for err in errors:
                st.error(err, icon="🚫")
        else:
            ok, message = update_expense(selected_id, title, amount, category,
                                         expense_date, payment_method, description)
            if ok:
                set_flash("success", message)
                st.rerun()
            else:
                st.error(message)


def render_delete_tab(df: pd.DataFrame):
    if df.empty:
        st.info("There are no expenses to delete.")
        return
    labels = expense_labels(df)
    selected_id = st.selectbox("Select an expense to delete", list(labels.keys()),
                               format_func=lambda i: labels[i])
    row = df[df["id"] == selected_id].iloc[0]
    st.warning(
        f"You are about to delete **{row['title']}** ({fmt_money(row['amount'])}, "
        f"{row['expense_date']:%Y-%m-%d}). This cannot be undone.",
        icon="⚠️",
    )
    confirm = st.checkbox("Yes, I am sure I want to delete this expense",
                          key=f"confirm_delete_{selected_id}")
    if st.button("🗑️ Delete Expense", type="primary", disabled=not confirm):
        ok, message = delete_expense(selected_id)
        set_flash("success" if ok else "error", message)
        st.rerun()


def page_expenses():
    page_header("📋 Expenses", "Search, filter, edit and delete your expenses")
    show_flash()
    df = get_expenses()
    tab_list, tab_edit, tab_delete = st.tabs(["📋 All Expenses", "✏️ Edit", "🗑️ Delete"])
    with tab_list:
        render_list_tab(df)
    with tab_edit:
        render_edit_tab(df)
    with tab_delete:
        render_delete_tab(df)


# =============================================================================
# 11. ANALYTICS (Matplotlib + Seaborn + Pandas)
# =============================================================================
def page_analytics():
    page_header("📊 Analytics", "Understand where your money goes")
    df = get_expenses()
    if df.empty:
        st.info("Add some expenses first to see charts.")
        return

    tabs = st.tabs(["By Category", "Monthly", "Trend", "Payment Methods", "Top Categories"])

    # 1. Expenses by category
    with tabs[0]:
        summary = get_category_summary(df)
        fig, ax = new_chart(8, max(3.5, 0.5 * len(summary) + 1.5))
        sns.barplot(data=summary, x="total", y="category", color="#4F46E5", ax=ax)
        label_horizontal_bars(ax, summary["total"].tolist())
        ax.set_xlabel("Total spent")
        ax.set_ylabel("")
        finish_chart(fig, "Expenses by Category", ax)
        with st.expander("View data"):
            show_table(summary)

    # 2. Monthly expenses
    with tabs[1]:
        monthly = get_monthly_summary(df)
        fig, ax = new_chart()
        sns.barplot(data=monthly, x="month", y="total", color="#7C3AED", ax=ax)
        symbol = st.session_state.get("currency", "₹")
        for i, v in enumerate(monthly["total"].tolist()):
            ax.text(i, v, f"{symbol}{v:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Month")
        ax.set_ylabel("Total spent")
        if len(monthly) > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        finish_chart(fig, "Monthly Expenses", ax)
        with st.expander("View data"):
            show_table(monthly)

    # 3. Daily / weekly trend
    with tabs[2]:
        mode = st.radio("Show", ["Daily (last 60 days)", "Weekly (last 26 weeks)"], horizontal=True)
        indexed = df.set_index("expense_date")["amount"]
        if mode.startswith("Daily"):
            series = indexed.resample("D").sum().tail(60)
        else:
            series = indexed.resample("W").sum().tail(26)
        fig, ax = new_chart()
        sns.lineplot(x=series.index, y=series.values, marker="o", color="#4F46E5", ax=ax)
        ax.fill_between(series.index, series.values, alpha=0.15, color="#4F46E5")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total spent")
        fig.autofmt_xdate()
        finish_chart(fig, "Spending Trend", ax)

    # 4. Payment method distribution
    with tabs[3]:
        pay = get_summary_by(df, "payment_method")
        fig, ax = new_chart(6, 4.5)
        ax.pie(
            pay["total"],
            labels=pay["payment_method"],
            autopct="%1.0f%%",
            startangle=90,
            colors=sns.color_palette("pastel", len(pay)),
            wedgeprops={"width": 0.45, "edgecolor": "white"},
        )
        ax.axis("equal")
        finish_chart(fig, "Payment Method Distribution", ax)
        with st.expander("View data"):
            show_table(pay)

    # 5. Top spending categories
    with tabs[4]:
        top = get_category_summary(df).head(5)
        fig, ax = new_chart(8, 4)
        sns.barplot(data=top, x="total", y="category", hue="category",
                    palette="crest_r", legend=False, ax=ax)
        label_horizontal_bars(ax, top["total"].tolist())
        ax.set_xlabel("Total spent")
        ax.set_ylabel("")
        finish_chart(fig, "Top 5 Spending Categories", ax)


# =============================================================================
# 12. BUDGET
# =============================================================================
def page_budget():
    page_header("💰 Budget", "Set a monthly budget and track it")
    show_flash()
    df = get_expenses()
    budget = get_budget()
    spent = get_current_month_total(df)

    with st.form("budget_form"):
        new_budget = st.number_input(
            f"Monthly budget ({st.session_state.get('currency', '₹')})",
            min_value=0.0, value=float(budget), step=100.0, format="%.2f",
        )
        saved = st.form_submit_button("💾 Save Budget", type="primary")
    if saved:
        if set_setting("monthly_budget", round(new_budget, 2)):
            set_flash("success", "Budget saved.")
            st.rerun()

    st.subheader(f"{date.today():%B %Y}")
    remaining = budget - spent
    pct = (spent / budget * 100) if budget > 0 else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Monthly Budget", fmt_money(budget))
    c2.metric("💸 Amount Spent", fmt_money(spent))
    c3.metric("🟢 Remaining", fmt_money(remaining))
    c4.metric("📊 Budget Used", f"{pct:.1f}%")
    budget_status(spent, budget)

    with st.expander("This month's spending by category"):
        if df.empty:
            st.write("No data yet.")
        else:
            today = date.today()
            month_df = df[(df["expense_date"].dt.year == today.year)
                          & (df["expense_date"].dt.month == today.month)]
            if month_df.empty:
                st.write("No expenses this month yet.")
            else:
                show_table(get_category_summary(month_df))


# =============================================================================
# 13. SETTINGS
# =============================================================================
def page_settings():
    page_header("⚙️ Settings", "Currency, data export, sample data and more")
    show_flash()
    df = get_expenses()

    tab_general, tab_data = st.tabs(["General", "Data"])

    with tab_general:
        st.subheader("Currency")
        current = st.session_state.get("currency", "₹")
        index = CURRENCIES.index(current) if current in CURRENCIES else 0
        choice = st.selectbox("Currency symbol", CURRENCIES, index=index)
        if st.button("Save currency"):
            if set_setting("currency", choice):
                st.session_state["currency"] = choice
                set_flash("success", "Currency saved.")
                st.rerun()

        st.subheader("Database")
        st.write(f"**File:** `{DB_PATH.name}`")
        st.write(f"**Saved expenses:** {count_expenses()}")

    with tab_data:
        st.subheader("Export")
        if df.empty:
            st.info("Nothing to export yet.")
        else:
            export_df = df.copy()
            export_df["expense_date"] = export_df["expense_date"].dt.strftime("%Y-%m-%d")
            st.download_button("⬇️ Download all expenses (CSV)",
                               export_df.to_csv(index=False).encode("utf-8"),
                               file_name="expenses.csv", mime="text/csv")

        st.subheader("Sample data (for testing)")
        st.caption(f"Sample expenses have a description starting with {SAMPLE_TAG}.")
        s1, s2 = st.columns(2)
        if s1.button("➕ Load sample data"):
            ok, message = add_sample_data()
            set_flash("success" if ok else "error", message)
            st.rerun()
        if s2.button("➖ Remove sample data"):
            ok, message = remove_sample_data()
            set_flash("success" if ok else "error", message)
            st.rerun()

        with st.expander("🧨 Danger zone - delete ALL expenses"):
            st.warning("This permanently deletes every expense.")
            typed = st.text_input("Type DELETE to confirm")
            if st.button("Delete everything", disabled=(typed != "DELETE")):
                ok, message = delete_all_expenses()
                set_flash("success" if ok else "error", message)
                st.rerun()


# =============================================================================
# 14. MAIN NAVIGATION
# =============================================================================
PAGES = {
    "🏠 Dashboard": page_dashboard,
    "➕ Add Expense": page_add_expense,
    "📋 Expenses": page_expenses,
    "📊 Analytics": page_analytics,
    "💰 Budget": page_budget,
    "⚙️ Settings": page_settings,
}


def main():
    if not initialize_database():
        st.stop()

    # Load the currency once per browser session
    if "currency" not in st.session_state:
        st.session_state["currency"] = get_setting("currency", "₹")

    st.sidebar.title("💸 Expense Tracker")
    page = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(f"📦 {count_expenses()} expenses saved")

    PAGES[page]()  # run the function of the chosen page


main()