"""
GUI module for the Polish IT Job Market Analysis project.
Provides a Tkinter interface to fetch, filter, and visualize salary data.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker

from main import fetch_remote_jobs, clean_and_analyze_salaries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] GUI – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


class JobMarketApp:
    """Main GUI application class."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Polish IT Job Market Analyzer (Adzuna)")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TButton", font="Helvetica 10 bold", padding="6 6")
        style.configure("TLabel", font="Helvetica 10")
        style.configure("Treeview.Heading", font="Helvetica 10 bold")
        style.configure("Treeview", font="Helvetica 10", rowheight="25")
        style.configure("Header.TLabel", font="Helvetica 10 bold")

        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.filtered_df: pd.DataFrame = pd.DataFrame()

        self._build_ui()

    def _build_ui(self) -> None:
        """Build all UI components."""
        # Controls
        control_frame = ttk.Frame(self.root, padding="10 10")
        control_frame.pack(side="top", fill="x")

        self.btn_fetch = ttk.Button(control_frame, text="Fetch Live Data (API)", command=self._start_fetch_thread)
        self.btn_fetch.pack(side="left", padx="0 10")

        self.btn_hist = ttk.Button(control_frame, text="Overall Distribution", command=self._show_histogram, state="disabled")
        self.btn_hist.pack(side="left", padx="0 10")

        self.btn_boxplot = ttk.Button(control_frame, text="Tech Comparison", command=self._show_boxplot, state="disabled")
        self.btn_boxplot.pack(side="left", padx="0 10")

        self.btn_bar = ttk.Button(control_frame, text="Top Locations", command=self._show_bar_chart, state="disabled")
        self.btn_bar.pack(side="left")

        # Filters
        filter_frame = ttk.Frame(self.root, padding="10 5")
        filter_frame.pack(side="top", fill="x")

        ttk.Label(filter_frame, text="Search Position:", style="Header.TLabel").pack(side="left", padx="0 5")

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._apply_filters())
        self.entry_search = ttk.Entry(filter_frame, textvariable=self.search_var, width=30)
        self.entry_search.pack(side="left", padx="0 20")

        ttk.Label(filter_frame, text="Filter Location:", style="Header.TLabel").pack(side="left", padx="0 5")

        self.location_var = tk.StringVar()
        self.location_var.trace_add("write", lambda *args: self._apply_filters())
        self.combo_location = ttk.Combobox(filter_frame, textvariable=self.location_var, state="readonly", width=30)
        self.combo_location.pack(side="left")
        self.combo_location.set("All Locations")

        # Data table
        data_frame = ttk.Frame(self.root, padding="10 10")
        data_frame.pack(side="top", fill="both", expand=True)

        columns = ("company", "position", "location", "salary_min", "salary_max", "salary_avg")
        self.tree = ttk.Treeview(data_frame, columns=columns, show="headings")

        self.tree.heading("company", text="Company")
        self.tree.heading("position", text="Position")
        self.tree.heading("location", text="Location")
        self.tree.heading("salary_min", text="Min Salary")
        self.tree.heading("salary_max", text="Max Salary")
        self.tree.heading("salary_avg", text="Avg Salary")

        self.tree.column("company", width=150, anchor="w")
        self.tree.column("position", width=250, anchor="w")
        self.tree.column("location", width=150, anchor="w")
        self.tree.column("salary_min", width=100, anchor="e")
        self.tree.column("salary_max", width=100, anchor="e")
        self.tree.column("salary_avg", width=100, anchor="e")

        scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Status bar
        status_frame = ttk.Frame(self.root, padding="5 5")
        status_frame.pack(side="bottom", fill="x")

        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Click 'Fetch Live Data' to begin.")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font="Helvetica 9 italic")
        status_label.pack(side="left")

    def _start_fetch_thread(self) -> None:
        """Start data fetch in a background thread."""
        self._set_ui_state("disabled")
        self.status_var.set("Fetching live market data... this takes ~30 seconds (please wait)...")
        self.root.config(cursor="watch")
        self.tree.delete(*self.tree.get_children())

        thread = threading.Thread(target=self._fetch_data_worker, daemon=True)
        thread.start()

    def _fetch_data_worker(self) -> None:
        """Fetch and clean data, then update UI."""
        try:
            raw_data = fetch_remote_jobs()
            clean_data = clean_and_analyze_salaries(raw_data)
            self.root.after(0, self._on_fetch_success, clean_data)
        except Exception as exc:
            logger.error("Error in background thread: %s", exc)
            self.root.after(0, self._on_fetch_error, str(exc))

    def _on_fetch_success(self, df: pd.DataFrame) -> None:
        """Handle successful data fetch."""
        self.raw_df = df

        locations = sorted([loc for loc in df["location"].unique() if loc.strip()])
        self.combo_location["values"] = ["All Locations"] + locations
        self.combo_location.set("All Locations")
        self.search_var.set("")

        self._apply_filters()

        self.root.config(cursor="")
        self._set_ui_state("normal")
        self.status_var.set(f"Successfully loaded {len(df)} jobs.")

    def _on_fetch_error(self, error_msg: str) -> None:
        """Handle fetch error."""
        self.root.config(cursor="")
        self._set_ui_state("normal")
        self.status_var.set("Error occurred during data fetch.")
        messagebox.showerror("Data Fetch Error", f"An error occurred:\n\n{error_msg}")

    def _set_ui_state(self, state: str) -> None:
        """Enable or disable interactive UI components."""
        self.btn_fetch.config(state=state)
        self.btn_hist.config(state=state)
        self.btn_boxplot.config(state=state)
        self.btn_bar.config(state=state)
        self.entry_search.config(state=state)
        self.combo_location.config(state="readonly" if state == "normal" else "disabled")

    def _apply_filters(self) -> None:
        """Filter data based on user input."""
        if self.raw_df.empty:
            return

        df = self.raw_df.copy()
        search_term = self.search_var.get().strip().lower()
        selected_location = self.location_var.get()

        if search_term:
            df = df[df["position"].str.lower().str.contains(search_term, na=False)]

        if selected_location and selected_location != "All Locations":
            df = df[df["location"] == selected_location]

        self.filtered_df = df
        self._render_treeview()

    def _render_treeview(self) -> None:
        """Populate the table with filtered data."""
        self.tree.delete(*self.tree.get_children())

        for _, row in self.filtered_df.iterrows():
            self.tree.insert("", "end", values=(
                row["company"],
                row["position"],
                row["location"],
                f"{row['salary_min']:,.0f} zł",
                f"{row['salary_max']:,.0f} zł",
                f"{row['salary_avg']:,.0f} zł"
            ))

        self.status_var.set(f"Showing {len(self.filtered_df)} of {len(self.raw_df)} offers.")

    def _show_histogram(self) -> None:
        """Plot salary distribution histogram."""
        if self.filtered_df.empty:
            messagebox.showinfo("No Data", "No data available to plot.")
            return

        window = tk.Toplevel(self.root)
        window.title("Overall Salary Distribution")
        window.geometry("800x600")

        location = self.location_var.get()
        search_term = self.search_var.get().strip()

        plot_title = "Distribution of Annual IT Salaries"
        plot_title += f" in {location}" if location and location != "All Locations" else " in Poland"
        if search_term:
            plot_title += f"\n(Filtered by: '{search_term}')"

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        salaries = self.filtered_df["salary_avg"]
        median_salary = salaries.median()

        ax.hist(salaries, bins=20, color="#4C72B0", edgecolor="black", alpha=0.7)
        ax.axvline(median_salary, color="#C44E52", linestyle="dashed", linewidth=2, label=f"Median: {median_salary:,.0f} zł")

        ax.set_title(plot_title, fontsize=14, pad=15)
        ax.set_xlabel("Annual Salary (PLN)", fontsize=12)
        ax.set_ylabel("Number of Job Offers", fontsize=12)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} zł'))
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def _show_boxplot(self) -> None:
        """Plot salary comparison by technology."""
        if self.raw_df.empty:
            return

        window = tk.Toplevel(self.root)
        window.title("Tech Stack Comparison")
        window.geometry("800x600")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        techs = ["Python", "Java", "React", "DevOps"]
        plot_data = []
        labels = []

        for tech in techs:
            subset = self.raw_df[self.raw_df["position"].str.contains(tech, case=False, na=False)]
            if not subset.empty:
                plot_data.append(subset["salary_avg"])
                labels.append(tech)

        if not plot_data:
            messagebox.showinfo("No Data", "Not enough specific technology data.")
            window.destroy()
            return

        box = ax.boxplot(plot_data, patch_artist=True)
        ax.set_xticklabels(labels)

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title("Annual IT Salary Distribution by Technology", fontsize=14, pad=15)
        ax.set_ylabel("Annual Salary (PLN)", fontsize=12)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} zł'))
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def _show_bar_chart(self) -> None:
        """Plot top 5 locations by salary."""
        if self.raw_df.empty:
            return

        window = tk.Toplevel(self.root)
        window.title("Top Locations by Salary")
        window.geometry("800x600")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        df_geo = self.raw_df[self.raw_df["location"].str.strip() != ""]
        grouped = df_geo.groupby("location")["salary_avg"].agg(["median", "count"])
        grouped = grouped[grouped["count"] >= 3]

        if grouped.empty:
            messagebox.showinfo("No Data", "Not enough localized data.")
            window.destroy()
            return

        top_locations = grouped.sort_values(by="median", ascending=False).head(5)
        bars = ax.bar(top_locations.index, top_locations["median"], color="#55A868", edgecolor="black", alpha=0.7)

        ax.set_title("Top 5 Locations by Median Annual Salary", fontsize=14, pad=15)
        ax.set_ylabel("Median Annual Salary (PLN)", fontsize=12)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} zł'))
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:,.0f} zł',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = JobMarketApp(root)
    root.mainloop()