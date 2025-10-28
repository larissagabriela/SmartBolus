# SmartBolus — Geração Automática de Bólus Personalizado em Radioterapia

O *SmartBolus* é uma ferramenta em Python desenvolvida para a geração automatizada de bólus personalizados utilizados em planejamentos de radioterapia. O sistema processa arquivos DICOM-RTSTRUCT e imagens de tomografia computadorizada (CT), identificando estruturas de interesse (como PTV e BODY) e construindo um bólus adaptado à anatomia do paciente. O código implementa operações morfológicas 3D, suavização da base, remoção de irregularidades e integração direta ao formato DICOM para exportação da nova estrutura (RTSTRUCT atualizado). O projeto foi desenvolvido com foco em automação, reprodutibilidade e compatibilidade com fluxos clínicos de radioterapia, podendo ser aplicado em estudos, ensino e pesquisa na área de Física Médica e Oncologia Radioterápica.

# ⚙️ Principais funcionalidades:

\item Extração e manipulação de estruturas DICOM (RTSTRUCT);

\item Criação automática de bólus com cavidade adaptada ao PTV;

\item Ajuste da espessura e posição do bólus conforme critérios geométricos;

\item Suavização da base e fechamento morfológico 3D;

\item Exportação direta para RTSTRUCT com UID válidos;

\item Visualização opcional das máscaras 3D e cortes axiais.

🧠 Aplicações:

\item Ensino e pesquisa em radioterapia e física médica;

\item Desenvolvimento de protocolos personalizados de planejamento;

\item Simulações Monte Carlo e estudos de modulação de dose superficial.

📦 Dependências principais:

'pydicom', 'numpy', 'scikit-image', 'scipy', 'matplotlib'
