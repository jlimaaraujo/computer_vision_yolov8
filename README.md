# YOLOv8 - Visão Computacional

Este projeto implementa soluções de visão computacional utilizando o modelo YOLOv8 para detecção de pessoas em imagens e vídeos. O objetivo é explorar as capacidades do YOLOv8 em diferentes cenários, como detecção de pessoas em espaço comercial, etc.

## Estrutura do Projeto

```
computer_vision_yolov8/
├── output/              # output de análise de desempenho e métricas
├── config/              # Arquivos de configuração do modelo
├── models/              # Modelos YOLOv8
├── src/                 # Código-fonte principal 
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação do projeto
```

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/jlimaaraujo/computer_vision_yolov8.git
   cd computer_vision_yolov8
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Como Utilizar

1. Execute o script principal:
    ```sh
    python src/main.py
    ```

---

Desenvolvido por **João Araújo** - AOOP
