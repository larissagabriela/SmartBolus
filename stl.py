"""
Módulo para extração e conversão de estruturas DICOM RTSTRUCT em modelos 3D STL.

Este script permite isolar uma estrutura específica (ROI) de um arquivo RTSTRUCT,
reconstruí-la como máscara volumétrica binária e gerar seu modelo 3D correspondente
em formato STL, centralizado no ponto (0, 0, 0).
"""

import os
import numpy as np
import pydicom
from skimage import measure
import trimesh
from matplotlib.path import Path


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
    slices = [pydicom.dcmread(os.path.join(ct_folder, f)) for f in os.listdir(ct_folder) if f.endswith('.dcm')]
    slices.sort(key=lambda s: s.ImagePositionPatient[2])
    z_positions = np.array([s.ImagePositionPatient[2] for s in slices])

    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    shape = (len(slices), slices[0].Rows, slices[0].Columns)
    mask = np.zeros(shape, dtype=np.uint8)

    # Identificar índice da estrutura
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

                rr, cc = np.meshgrid(np.arange(slices[0].Rows), np.arange(slices[0].Columns), indexing='ij')
                poly = Path(np.vstack((col_coords, row_coords)).T)
                coords = np.vstack((cc.flatten(), rr.flatten())).T
                inside = poly.contains_points(coords)

                mask_slice = inside.reshape((slices[0].Rows, slices[0].Columns))
                mask[slice_idx][mask_slice] = 1

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

    print(f"✅ STL exportado para: {output_path}")
    print(f"📏 Modelo centralizado no ponto (0,0,0)")



def main():
    """Função principal de execução do script."""
    rtstruct_path = "RS.21_12_2015.FANTOMA_eletBOLUS.dcm"  # Caminho do RTSTRUCT
    ct_folder = "pasta_tc"                                 # Pasta com os DICOMs de CT
    structure_name = "BolusECT_v9"                         # Nome da estrutura desejada
    output_stl = f"{structure_name}.stl"

    rtstruct = load_rtstruct(rtstruct_path)
    mask, spacing, dz = extract_structure_mask(rtstruct, ct_folder, structure_name)
    mask_to_stl(mask, spacing, dz, output_stl)


if __name__ == "__main__":
    main()