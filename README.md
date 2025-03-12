# Streamlit Project

## Project Setup

### Prerequisites
- Python 3.10.11
- pip
- virtualenv

### Installation Steps
1. Clone the repository
2. Create a virtual environment
```bash
# Your current versions
python -m venv venv
# DEFINE VERSIONS
py -3.10 -m venv venv 
```
3. Activate virtual environment
```bash
# On Window
venv\Scripts\activate
# On Linux OR MacOS
source venv/bin/activate
```
4. Install dependencies on `requirements.txt` or `setup.py` (Optional)
```bash
pip install -r requirements.txt # Installs from requirements.txt
pip install -e .  # Installs from setup.py
```
5. Run the Streamlit
```bash
streamlit run src/main.py
```
