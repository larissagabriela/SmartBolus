"""
Módulo para extração e conversão de estruturas DICOM RTSTRUCT em modelos 3D STL com interface gráfica.

Este programa permite que o usuário selecione um arquivo RTSTRUCT, a pasta contendo as imagens de CT
correspondentes e o nome da estrutura (ROI) desejada para reconstrução. A partir disso, o script
gera automaticamente um modelo 3D em formato STL, centralizado no ponto (0, 0, 0).

A interface gráfica é baseada em Tkinter, biblioteca nativa e gratuita do Python, garantindo portabilidade
sem dependências externas.  
"""

import os
import numpy as np
import pydicom
from skimage import measure
from matplotlib.path import Path
import trimesh
import threading
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox


# ============================
# 🔹 Funções principais (núcleo de processamento)
# ============================

def load_rtstruct(rtstruct_path: str) -> pydicom.dataset.FileDataset:
    """Carrega o arquivo RTSTRUCT especificado.

    Parâmetros:
        rtstruct_path (str): Caminho completo do arquivo RTSTRUCT (.dcm).

    Retorna:
        FileDataset: Objeto DICOM contendo o conjunto de estruturas (RTSTRUCT).
    """
    return pydicom.dcmread(rtstruct_path)


def get_structure_names(rtstruct: pydicom.dataset.FileDataset) -> list[str]:
    """Lista os nomes de todas as estruturas (ROIs) contidas em um RTSTRUCT.

    Parâmetros:
        rtstruct (FileDataset): Objeto DICOM RTSTRUCT carregado.

    Retorna:
        list[str]: Lista de nomes das estruturas disponíveis.
    """
    return [roi.ROIName for roi in rtstruct.StructureSetROISequence]


def extract_structure_mask(rtstruct: pydicom.dataset.FileDataset, ct_folder: str, structure_name: str):
    """Constrói uma máscara binária 3D para a estrutura especificada.

    Esta função converte os contornos (ContourData) de uma estrutura do RTSTRUCT em uma
    matriz volumétrica binária, alinhada com as imagens de tomografia (CT) de referência.

    Parâmetros:
        rtstruct (FileDataset): Arquivo RTSTRUCT carregado.
        ct_folder (str): Pasta contendo os arquivos DICOM das imagens de CT.
        structure_name (str): Nome da estrutura (ROI) a ser extraída.

    Retorna:
        tuple[np.ndarray, list[float], float]:
            - Máscara binária 3D (z, y, x)
            - Espaçamento dos pixels (mm)
            - Espessura dos cortes (mm)
    """
    # Leitura e ordenação das fatias da tomografia
    slices = [pydicom.dcmread(os.path.join(ct_folder, f))
              for f in os.listdir(ct_folder) if f.endswith('.dcm')]
    slices.sort(key=lambda s: s.ImagePositionPatient[2])
    z_positions = np.array([s.ImagePositionPatient[2] for s in slices])

    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = abs(z_positions[1] - z_positions[0])
    mask = np.zeros((len(slices), slices[0].Rows, slices[0].Columns), dtype=np.uint8)

    # Identificar índice da estrutura desejada
    structure_index = None
    for roi in rtstruct.StructureSetROISequence:
        if roi.ROIName.lower() == structure_name.lower():
            structure_index = roi.ROINumber
            break
    if structure_index is None:
        raise ValueError(f"Estrutura '{structure_name}' não encontrada no RTSTRUCT.")

    # Construção da máscara voxel a voxel
    for roi_contour in rtstruct.ROIContourSequence:
        if roi_contour.ReferencedROINumber == structure_index:
            for contour in roi_contour.ContourSequence:
                pts = np.array(contour.ContourData).reshape(-1, 3)
                z = pts[0, 2]
                slice_idx = np.argmin(np.abs(z_positions - z))

                row_coords = (pts[:, 1] - slices[0].ImagePositionPatient[1]) / pixel_spacing[1]
                col_coords = (pts[:, 0] - slices[0].ImagePositionPatient[0]) / pixel_spacing[0]

                rr, cc = np.meshgrid(np.arange(slices[0].Rows),
                                     np.arange(slices[0].Columns),
                                     indexing='ij')
                poly = Path(np.vstack((col_coords, row_coords)).T)
                inside = poly.contains_points(np.vstack((cc.flatten(), rr.flatten())).T)

                mask[slice_idx][inside.reshape(mask[slice_idx].shape)] = 1

    return mask, pixel_spacing, slice_thickness


def mask_to_stl(mask: np.ndarray, pixel_spacing: list[float], slice_thickness: float, output_path: str) -> None:
    """Gera um arquivo STL 3D a partir de uma máscara volumétrica binária.

    A superfície é reconstruída via o algoritmo Marching Cubes e centralizada
    geometricamente em torno da origem (0,0,0).

    Parâmetros:
        mask (np.ndarray): Máscara binária tridimensional (z, y, x).
        pixel_spacing (list[float]): Espaçamento dos pixels [mm] em X e Y.
        slice_thickness (float): Espessura dos cortes (mm).
        output_path (str): Caminho de saída do arquivo STL.
    """
    verts, faces, normals, _ = measure.marching_cubes(
        mask, level=0.5, spacing=(slice_thickness, pixel_spacing[0], pixel_spacing[1])
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.apply_translation(-mesh.centroid)
    mesh.export(output_path)


# Funções da interface gráfica (Tkinter)

def log(msg: str) -> None:
    """Registra mensagens no campo de log da interface."""
    text_log.config(state="normal")
    text_log.insert(tk.END, msg + "\n")
    text_log.see(tk.END)
    text_log.config(state="disabled")


def select_rt() -> None:
    """Seleciona o arquivo RTSTRUCT via diálogo."""
    path = filedialog.askopenfilename(title="Selecionar RTSTRUCT (.dcm)", filetypes=[("DICOM", "*.dcm")])
    if path:
        entry_rt.delete(0, tk.END)
        entry_rt.insert(0, path)


def select_ct_folder() -> None:
    """Seleciona a pasta contendo as imagens de tomografia."""
    folder = filedialog.askdirectory(title="Selecionar pasta de CTs")
    if folder:
        entry_ct.delete(0, tk.END)
        entry_ct.insert(0, folder)


def select_output() -> None:
    """Seleciona o caminho e nome do arquivo STL de saída."""
    path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL", "*.stl")])
    if path:
        entry_out.delete(0, tk.END)
        entry_out.insert(0, path)


def load_structures() -> None:
    """Carrega as estruturas disponíveis no RTSTRUCT e atualiza a lista na interface."""
    try:
        rt = load_rtstruct(entry_rt.get())
        names = get_structure_names(rt)
        combo_struct["values"] = names
        combo_struct.set(names[0] if names else "")
        log(f"Estruturas carregadas ({len(names)} encontradas)")
    except Exception as e:
        log(f"Erro ao ler RTSTRUCT: {e}")


def run_conversion() -> None:
    """Inicia o processamento em thread separada (mantém a interface responsiva)."""
    threading.Thread(target=process, daemon=True).start()


def process() -> None:
    """Executa o fluxo principal de conversão: RTSTRUCT → máscara → STL."""
    try:
        rt_path = entry_rt.get()
        ct_folder = entry_ct.get()
        struct_name = combo_struct.get()
        out_path = entry_out.get()

        if not all([rt_path, ct_folder, struct_name, out_path]):
            raise ValueError("Preencha todos os campos antes de executar.")

        log("Processando...")
        rt = load_rtstruct(rt_path)
        mask, spacing, dz = extract_structure_mask(rt, ct_folder, struct_name)
        mask_to_stl(mask, spacing, dz, out_path)
        log(f"STL gerado com sucesso: {out_path}")
        messagebox.showinfo("Concluído", f"STL salvo em:\n{out_path}")
    except Exception as e:
        log(f"Erro: {e}")
        messagebox.showerror("Erro", str(e))


# Construção da janela principal

root = tk.Tk()
root.title("Conversor RTSTRUCT → STL")
root.geometry("640x400")
root.resizable(False, False)

frm = ttk.Frame(root, padding=10)
frm.pack(fill="both", expand=True)

# Campos de entrada e botões
ttk.Label(frm, text="RTSTRUCT (.dcm):").grid(row=0, column=0, sticky="w")
entry_rt = ttk.Entry(frm, width=50)
entry_rt.grid(row=0, column=1, padx=5)
ttk.Button(frm, text="Procurar", command=select_rt).grid(row=0, column=2)

ttk.Label(frm, text="Pasta CT:").grid(row=1, column=0, sticky="w")
entry_ct = ttk.Entry(frm, width=50)
entry_ct.grid(row=1, column=1, padx=5)
ttk.Button(frm, text="Procurar", command=select_ct_folder).grid(row=1, column=2)

ttk.Button(frm, text="Ler estruturas", command=load_structures).grid(row=2, column=0, pady=5)
combo_struct = ttk.Combobox(frm, width=47, state="readonly")
combo_struct.grid(row=2, column=1, columnspan=2, sticky="w")

ttk.Label(frm, text="Salvar STL como:").grid(row=3, column=0, sticky="w")
entry_out = ttk.Entry(frm, width=50)
entry_out.grid(row=3, column=1, padx=5)
ttk.Button(frm, text="Procurar", command=select_output).grid(row=3, column=2)

ttk.Button(frm, text="Executar", command=run_conversion).grid(row=4, column=1, pady=10, sticky="e")

# Campo de log
text_log = scrolledtext.ScrolledText(frm, width=75, height=10, state="disabled")
text_log.grid(row=5, column=0, columnspan=3, pady=5)

root.mainloop()
