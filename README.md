# Installing Dependencies from requirements.txt

## Prerequisites
- Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- Ensure you have `pip` installed. It usually comes with Python, but you can check by running:
  ```sh
  python -m pip --version
  ```

## Installation Steps

1. **Navigate to the Project Directory**
   Open a terminal or command prompt and navigate to the directory where `requirements.txt` is located:
   ```sh
   cd /path/to/project
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   It's best practice to install dependencies in a virtual environment to avoid conflicts:
   ```sh
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

3. **Install Dependencies**
   Run the following command to install all dependencies listed in `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   To check if the dependencies were installed correctly, run:
   ```sh
   pip list
   ```

## Troubleshooting
- If you encounter permission issues, try running:
  ```sh
  pip install --user -r requirements.txt
  ```
- If installation fails due to outdated `pip`, upgrade it with:
  ```sh
  pip install --upgrade pip
  ```

## Deactivating the Virtual Environment
If you used a virtual environment, you can deactivate it by running:
```sh
 deactivate
```

