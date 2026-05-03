# hackathon-benders

## Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda ayağa kaldırmak için aşağıdaki adımları izleyin:

### Sanal Ortam Oluşturma ve Api Çalıştırma
```bash
git clone https://github.com/akbulute/hackathon-benders.git
cd hackathon-benders

python -m venv .venv
.venv/Scripts/activate #.venv/bin/activate for linux

pip install fastapi uvicorn ortools pandas pydantic
uvicorn api:app --reload --port 8000
```
HTML dosyasını çalıştırdıktan sonra projeyi deneyebilirsiniz.

Sunum Drive Linki : 
