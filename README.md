# AutoNav

### A lane and object detection model for self driving cars

---

## 1. Setup

1. Clone the repository using:

   ```shell
   git clone git@github.com:Sailesh-2209/autonav.git
   ```

2. Create a virtual environment that uses `Python 3.7` named venv using the command:

   ```shell
   virtualenv --python=<py-3.7_path> venv
   ```

   Replace <py-3.7_path> with the path to python.exe (version 3.7)

3. Install the required dependencies using the requirements.txt file.

   ```shell
   pip install -r requirements.txt
   ```

### Installing CARLA

1. This project uses version `0.9.10` of the carla simulator.
   Download [here](https://github.com/carla-simulator/carla/releases/tag/0.9.10)
2. In the `test.py` file, change the variable `CARLA_PATH` to match the path of the egg file in the
   `PythonAPI/carla/dist` folder in the root of the carla installation folder.
