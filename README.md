# Setup and run instructions
1. Install python 3.10
2. Install `python3-venv` on Ubuntu or equivalent for your OS
3. Create virtual environment from inside project directory
```$ python -m venv venv```
4. `source ./venv/bin/activate` or equivalent for your OS
5. To run the python file directly as an executable: `$ chmod u+x tetris.py`
6. If running as script:
    `$ ./tetris.py < input.txt > output.txt`
    or if running by explicitly invoking the interpreter:
    `$ python tetris.py < input.txt > output.txt`