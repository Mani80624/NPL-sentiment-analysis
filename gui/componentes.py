import tkinter as tk
from tkinter import scrolledtext

from gui.estilos import (
    COLOR_PANEL_IZQUIERDO,
    COLOR_TITULO,
    COLOR_TEXTO_OSCURO,
    COLOR_BOTON_PRIMARIO,
    COLOR_BOTON_PRIMARIO_HOVER,
    COLOR_BOTON_LIMPIAR,
    COLOR_BOTON_LIMPIAR_HOVER,
    COLOR_CAJA_TEXTO,
    FUENTE_TITULO,
    FUENTE_NORMAL,
    ANCHO_BOTON,
    ALTO_BOTON
)


def crear_titulo(master, texto):
    return tk.Label(
        master,
        text=texto,
        font=FUENTE_TITULO,
        bg=master.cget("bg"),
        fg=COLOR_TITULO
    )


def crear_boton(master, texto, comando, tipo="primario"):
    if tipo == "limpiar":
        color_normal = COLOR_BOTON_LIMPIAR
        color_hover = COLOR_BOTON_LIMPIAR_HOVER
    else:
        color_normal = COLOR_BOTON_PRIMARIO
        color_hover = COLOR_BOTON_PRIMARIO_HOVER

    boton = tk.Button(
        master,
        text=texto,
        command=comando,
        font=FUENTE_NORMAL,
        bg=color_normal,
        fg=COLOR_TEXTO_OSCURO,
        activebackground=color_hover,
        activeforeground=COLOR_TEXTO_OSCURO,
        relief="flat",
        bd=0,
        cursor="hand2",
        width=ANCHO_BOTON,
        height=ALTO_BOTON,
        padx=8,
        pady=4
    )

    boton.bind("<Enter>", lambda event: boton.config(bg=color_hover))
    boton.bind("<Leave>", lambda event: boton.config(bg=color_normal))

    return boton


def crear_texto_scroll(master, alto=10, ancho=60):
    return scrolledtext.ScrolledText(
        master,
        height=alto,
        width=ancho,
        font=FUENTE_NORMAL,
        bg=COLOR_CAJA_TEXTO,
        fg=COLOR_TEXTO_OSCURO,
        insertbackground=COLOR_TEXTO_OSCURO,
        wrap="word",
        relief="solid",
        bd=1
    )


def crear_panel(master, color_fondo):
    return tk.Frame(
        master,
        bg=color_fondo,
        bd=0,
        highlightthickness=0
    )