"""
src/gui.py
----------
Graphical User Interface for the Polish IT Job Market Analysis project.

This module provides a modern Tkinter interface to asynchronously fetch
salary data from the Adzuna API (via main.py) without freezing the main
thread. It displays the cleaned data in a sortable Treeview and embeds
a Matplotlib histogram to visualize the salary distribution.

Usage:
    python src/gui.py
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker

# Import the core logic from our robust backend
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
        self.root.geometry("1000x600")
        self.root.minsize(800, 500)

        # Apply a modern, clean theme
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Configure styles (using strings to avoid Python 3.14 TclError bugs)
        style.configure("TButton", font="Helvetica 10 bold", padding="6 6")
        style.configure("TLabel", font="Helvetica 10")
        style.configure("Treeview.Heading", font="Helvetica 10 bold")
        style.configure("Treeview", font="Helvetica 10", rowheight="25")

        # Hold the data in memory
        self.current_df: pd.DataFrame | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the main frames, buttons, and treeview."""
        # --- Top Control Frame ---
        control_frame = ttk.Frame(self.root, padding="10 10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_fetch = ttk.Button(
            control_frame,
            text="Fetch Live Data (API)",
            command=self._start_fetch_thread
        )
        self.btn_fetch.pack(side=tk.LEFT, padx="0 10")

        self.btn_plot = ttk.Button(
            control_frame,
            text="Show Salary Distribution",
            command=self._show_plot,
            state=tk.DISABLED
        )
        self.btn_plot.pack(side=tk.LEFT)

        # --- Middle Data Frame (Treeview) ---
        data_frame = ttk.Frame(self.root, padding="10 0")
        data_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columns = ("company", "position", "location", "salary_min", "salary_max", "salary_avg")
        self.tree = ttk.Treeview(data_frame, columns=columns, show="headings")

        # Define headings
        self.tree.heading("company", text="Company")
        self.tree.heading("position", text="Position")
        self.tree.heading("location", text="Location")
        self.tree.heading("salary_min", text="Min Salary (PLN)")
        self.tree.heading("salary_max", text="Max Salary (PLN)")
        self.tree.heading("salary_avg", text="Avg Salary (PLN)")

        # Define column widths
        self.tree.column("company", width=150, anchor=tk.W)
        self.tree.column("position", width=250, anchor=tk.W)
        self.tree.column("location", width=150, anchor=tk.W)
        self.tree.column("salary_min", width=100, anchor=tk.E)
        self.tree.column("salary_max", width=100, anchor=tk.E)
        self.tree.column("salary_avg", width=100, anchor=tk.E)

        # Scrollbar
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Bottom Status Frame ---
        status_frame = ttk.Frame(self.root, padding="5 5")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Click 'Fetch Live Data' to begin.")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font="Helvetica 9 italic")
        status_label.pack(side=tk.LEFT)

    def _start_fetch_thread(self) -> None:
        """Lock the UI and start the data ingestion background thread."""
        self.btn_fetch.config(state=tk.DISABLED)
        self.btn_plot.config(state=tk.DISABLED)
        self.status_var.set("Fetching live market data... this takes ~30 seconds (please wait)...")
        self.root.config(cursor="watch")

        # Clear existing data in the tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        logger.info("Starting background thread for API fetch.")
        thread = threading.Thread(target=self._fetch_data_worker, daemon=True)
        thread.start()

    def _fetch_data_worker(self) -> None:
        """Background thread executing the heavy API requests and Pandas cleaning."""
        try:
            raw_df = fetch_remote_jobs()
            clean_df = clean_and_analyze_salaries(raw_df)

            # Safely schedule the UI update back on the main thread
            self.root.after(0, self._on_fetch_success, clean_df)

        except Exception as exc:
            logger.error("Error in background thread: %s", exc)
            self.root.after(0, self._on_fetch_error, str(exc))

    def _on_fetch_success(self, df: pd.DataFrame) -> None:
        """Callback executed on the main thread when data is ready."""
        self.current_df = df

        # Populate the Treeview
        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=(
                row["company"],
                row["position"],
                row["location"],
                f"{row['salary_min']:,.0f}",
                f"{row['salary_max']:,.0f}",
                f"{row['salary_avg']:,.0f}"
            ))

        # Restore UI state
        self.root.config(cursor="")
        self.btn_fetch.config(state=tk.NORMAL)
        self.btn_plot.config(state=tk.NORMAL)

        stats = f"Successfully loaded {len(df)} jobs with transparent salary data."
        self.status_var.set(stats)
        logger.info(stats)

    def _on_fetch_error(self, error_msg: str) -> None:
        """Callback executed on the main thread if the background thread fails."""
        self.root.config(cursor="")
        self.btn_fetch.config(state=tk.NORMAL)
        self.status_var.set("Error occurred during data fetch.")
        messagebox.showerror("Data Fetch Error", f"An error occurred:\n\n{error_msg}")

    def _show_plot(self) -> None:
        """Open a new Toplevel window and draw the Matplotlib histogram."""
        if self.current_df is None or self.current_df.empty:
            messagebox.showinfo("No Data", "Please fetch data first.")
            return

        plot_window = tk.Toplevel(self.root)
        plot_window.title("Polish IT Salary Distribution")
        plot_window.geometry("800x600")

        # Create Matplotlib Figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        salaries = self.current_df["salary_avg"]
        median_salary = salaries.median()

        # Draw Histogram
        n, bins, patches = ax.hist(
            salaries,
            bins=20,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.7
        )

        # Draw Median Line
        ax.axvline(
            median_salary,
            color="#C44E52",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median_salary:,.0f} PLN"
        )

        # Styling
        ax.set_title("Distribution of Annual IT Salaries in Poland", fontsize=14, pad=15)
        ax.set_xlabel("Annual Salary (PLN)", fontsize=12)
        ax.set_ylabel("Number of Job Offers", fontsize=12)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} zł'))
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()

        fig.tight_layout()

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = JobMarketApp(root)
    root.mainloop()