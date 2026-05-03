# hackathon-benders

## Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda ayağa kaldırmak için aşağıdaki adımları izleyin:

### Gereksinimleri Yükleme ve Çalıştırma
```bash
git clone https://github.com/akbulute/hackathon-benders.git
cd hackathon-benders
pip install fastapi uvicorn ortools pandas pydantic
uvicorn api:app --reload --port 8000
