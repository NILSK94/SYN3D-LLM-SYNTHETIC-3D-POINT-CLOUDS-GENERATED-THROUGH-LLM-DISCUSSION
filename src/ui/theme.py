import tkinter as tk
from tkinter import ttk

def apply_dark_mode(root: tk.Tk):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    bg = "#0b0c10"
    fg = "#e8e8ea"
    panel = "#111218"
    panel2 = "#151725"
    border = "#25263a"
    accent = "#1f2333"
    accent2 = "#2a2f46"
    select_bg = "#2a4b8d"

    root.configure(bg=bg)

    default_font = ("Segoe UI", 10)
    title_font = ("Segoe UI", 11, "bold")

    style.configure(".", background=bg, foreground=fg, fieldbackground=panel, bordercolor=border, font=default_font)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg, font=default_font)
    style.configure("Title.TLabel", background=bg, foreground=fg, font=title_font)
    style.configure("TLabelFrame", background=bg, foreground=fg)
    style.configure("TLabelFrame.Label", background=bg, foreground=fg, font=title_font)

    style.configure("TCheckbutton", background=bg, foreground=fg)
    style.map("TCheckbutton", background=[("active", bg)], foreground=[("active", fg)])

    style.configure("TButton", background=accent2, foreground=fg, borderwidth=1, focusthickness=1, focuscolor=border)
    style.map("TButton",
              background=[("active", accent), ("disabled", bg)],
              foreground=[("disabled", "#7a7a7a")])

    style.configure("TEntry", fieldbackground=panel2, foreground=fg, insertcolor=fg)
    style.configure("TCombobox", fieldbackground=panel2, foreground=fg, background=panel2, arrowcolor=fg)
    style.map("TCombobox",
              fieldbackground=[("readonly", panel2), ("active", panel)],
              background=[("readonly", panel2), ("active", panel)],
              foreground=[("readonly", fg)])

    style.configure("TNotebook", background=bg, borderwidth=0)
    style.configure("TNotebook.Tab", background=panel, foreground=fg, padding=(14, 8))
    style.map("TNotebook.Tab",
              background=[("selected", panel2), ("active", panel)],
              foreground=[("selected", fg), ("active", fg)])

    def config_text_widget(w: tk.Text):
        w.configure(
            bg=panel2,
            fg=fg,
            insertbackground=fg,
            selectbackground=select_bg,
            selectforeground=fg,
            highlightbackground=border,
            highlightcolor=border,
            relief=tk.FLAT,
            font=default_font,
        )
    return config_text_widget, (bg, fg, panel2, border)
