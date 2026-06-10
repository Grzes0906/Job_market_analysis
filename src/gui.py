"""
src/gui.py
----------
Graphical User Interface for the Polish IT Job Market Analysis project.

This module provides a modern Tkinter interface to asynchronously fetch
salary data from the Adzuna API (via main.py) without freezing the main thread.
It implements interactive Pandas-based filtering and embeds advanced Matplotlib
visualizations (Histogram, Boxplot, Bar Chart) to satisfy academic requirements.

Usage:
    python src/gui.py
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

# Import the core logic from our backend
from main import fetch_remote_jobs, clean_and_analyze_salaries

# Configure local logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] GUI – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


class JobMarketApp:
    """Main application class for the Job Market Analyzer."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Polish IT Job Market Analyzer (Adzuna)")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        # Apply a modern, clean theme
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Configure styles using pure strings to satisfy PyCharm strict typing
        style.configure("TButton", font="Helvetica 10 bold", padding="6 6")
        style.configure("TLabel", font="Helvetica 10")
        style.configure("Treeview.Heading", font="Helvetica 10 bold")
        style.configure("Treeview", font="Helvetica 10", rowheight="25")
        style.configure("Header.TLabel", font="Helvetica 10 bold")

        # Data holding structures
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.filtered_df: pd.DataFrame = pd.DataFrame()

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the main frames, buttons, filters, and treeview."""
        # -------------------------------------------------------------------
        # 1. Top Control Frame
        # -------------------------------------------------------------------
        control_frame = ttk.Frame(self.root, padding="10 10")
        control_frame.pack(side="top", fill="x")

        self.btn_fetch = ttk.Button(
            control_frame,
            text="Fetch Live Data (API)",
            command=self._start_fetch_thread
        )
        self.btn_fetch.pack(side="left", padx="0 10")

        self.btn_hist = ttk.Button(
            control_frame,
            text="Overall Distribution",
            command=self._show_histogram,
            state="disabled"
        )
        self.btn_hist.pack(side="left", padx="0 10")

        self.btn_boxplot = ttk.Button(
            control_frame,
            text="Tech Comparison",
            command=self._show_boxplot,
            state="disabled"
        )
        self.btn_boxplot.pack(side="left", padx="0 10")

        self.btn_bar = ttk.Button(
            control_frame,
            text="Top Locations",
            command=self._show_bar_chart,
            state="disabled"
        )
        self.btn_bar.pack(side="left")

        # -------------------------------------------------------------------
        # 2. Interactive Filtering Frame
        # -------------------------------------------------------------------
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
        self.combo_location = ttk.Combobox(
            filter_frame,
            textvariable=self.location_var,
            state="readonly",
            width=30
        )
        self.combo_location.pack(side="left")
        self.combo_location.set("All Locations")

        # -------------------------------------------------------------------
        # 3. Middle Data Frame (Treeview)
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # 4. Bottom Status Frame
        # -------------------------------------------------------------------
        status_frame = ttk.Frame(self.root, padding="5 5")
        status_frame.pack(side="bottom", fill="x")

        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Click 'Fetch Live Data' to begin.")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font="Helvetica 9 italic")
        status_label.pack(side="left")

    # =======================================================================
    # Threading & API Integration
    # =======================================================================

    def _start_fetch_thread(self) -> None:
        """Lock the UI and start the data ingestion background thread."""
        self._set_ui_state("disabled")
        self.status_var.set("Fetching live market data... this takes ~30 seconds (please wait)...")
        self.root.config(cursor="watch")

        self.tree.delete(*self.tree.get_children())

        logger.info("Starting background thread for API fetch.")
        thread = threading.Thread(target=self._fetch_data_worker, daemon=True)
        thread.start()

    def _fetch_data_worker(self) -> None:
        """Background thread executing the heavy API requests and cleaning."""
        try:
            raw_data = fetch_remote_jobs()
            clean_data = clean_and_analyze_salaries(raw_data)

            # Safely schedule the UI update back on the main Tkinter thread
            self.root.after(0, self._on_fetch_success, clean_data)

        except Exception as exc:
            logger.error("Error in background thread: %s", exc)
            self.root.after(0, self._on_fetch_error, str(exc))

    def _on_fetch_success(self, df: pd.DataFrame) -> None:
        """Callback executed on the main thread when data is ready."""
        self.raw_df = df

        # Populate Location Combobox
        locations = sorted([loc for loc in df["location"].unique() if loc.strip()])
        self.combo_location["values"] = ["All Locations"] + locations
        self.combo_location.set("All Locations")
        self.search_var.set("")

        self._apply_filters()

        self.root.config(cursor="")
        self._set_ui_state("normal")

        stats = f"Successfully loaded {len(df)} jobs with transparent salary data."
        self.status_var.set(stats)
        logger.info(stats)

    def _on_fetch_error(self, error_msg: str) -> None:
        """Callback executed on the main thread if the background thread fails."""
        self.root.config(cursor="")
        self._set_ui_state("normal")
        self.status_var.set("Error occurred during data fetch.")
        messagebox.showerror("Data Fetch Error", f"An error occurred:\n\n{error_msg}")

    def _set_ui_state(self, state: str) -> None:
        """Helper to bulk-enable/disable interactive components."""
        self.btn_fetch.config(state=state)
        self.btn_hist.config(state=state)
        self.btn_boxplot.config(state=state)
        self.btn_bar.config(state=state)
        self.entry_search.config(state=state)

        combo_state = "readonly" if state == "normal" else "disabled"
        self.combo_location.config(state=combo_state)

    # =======================================================================
    # Vectorized Filtering & Rendering
    # =======================================================================

    def _apply_filters(self) -> None:
        """Filter the dataframe dynamically based on Entry and Combobox inputs."""
        if self.raw_df.empty:
            return

        df = self.raw_df.copy()
        search_term = self.search_var.get().strip().lower()
        selected_location = self.location_var.get()

        if search_term:
            # Case-insensitive Pandas text filter
            df = df[df["position"].str.lower().str.contains(search_term, na=False)]

        if selected_location and selected_location != "All Locations":
            df = df[df["location"] == selected_location]

        self.filtered_df = df
        self._render_treeview()

    def _render_treeview(self) -> None:
        """Clear and populate the Treeview with formatted data."""
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

        self.status_var.set(f"Showing {len(self.filtered_df)} of {len(self.raw_df)} total offers.")

    # =======================================================================
    # Matplotlib Visualizations
    # =======================================================================

    def _show_histogram(self) -> None:
        """Draw a histogram of overall salary distribution based on current filters."""
        if self.filtered_df.empty:
            messagebox.showinfo("No Data", "No data available to plot.")
            return

        window = tk.Toplevel(self.root)
        window.title("Overall Salary Distribution")
        window.geometry("800x600")

        # --- DYNAMICZNY TYTUŁ WYKRESU ---
        location = self.location_var.get()
        search_term = self.search_var.get().strip()

        plot_title = "Distribution of Annual IT Salaries"
        if location and location != "All Locations":
            plot_title += f" in {location}"
        else:
            plot_title += " in Poland"

        if search_term:
            plot_title += f"\n(Filtered by: '{search_term}')"
        # --------------------------------

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        salaries = self.filtered_df["salary_avg"]
        median_salary = salaries.median()

        ax.hist(salaries, bins=20, color="#4C72B0", edgecolor="black", alpha=0.7)
        ax.axvline(
            median_salary,
            color="#C44E52",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median_salary:,.0f} zł"
        )

        # Wstrzykujemy nasz dynamiczny tekst do matplotlib
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
        """Draw a boxplot comparing different technology stacks."""
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
            messagebox.showinfo("No Data", "Not enough specific technology data to compare.")
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
        """Draw a bar chart of top-paying geographical locations."""
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
            messagebox.showinfo("No Data", "Not enough localized data available (min 3 offers required).")
            window.destroy()
            return

        top_locations = grouped.sort_values(by="median", ascending=False).head(5)

        bars = ax.bar(
            top_locations.index,
            top_locations["median"],
            color="#55A868",
            edgecolor="black",
            alpha=0.7
        )

        ax.set_title("Top 5 Locations by Median Annual Salary", fontsize=14, pad=15)
        ax.set_ylabel("Median Annual Salary (PLN)", fontsize=12)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} zł'))
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,.0f} zł',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = JobMarketApp(root)
    root.mainloop()