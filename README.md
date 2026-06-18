## Uma IA utilizando mais de 100mil partidas de touhou 12.3

link para baixar o dataset:
https://drive.google.com/file/d/1CdNEDXCzDQ3fsvZsPj6d4IBTW1wvcETL/view?usp=drive_link

## Interface Web

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute a aplicação:
   ```bash
   python app.py
   ```
3. Acesse no navegador:
   ```text
   http://127.0.0.1:5000/
   ```

A aplicação oferece um formulário web para enviar ELO e personagens e receber a previsão do vencedor via `/api/predict`.


##comandos para criar o ambiente virtual 
python3.14 -m venv .venv 
source .venv/bin/activate
