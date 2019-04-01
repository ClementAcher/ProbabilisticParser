# Parser

You may need to start by make the shell file executable :

```
chmod +x run.sh
```

Then you can try out the parser in two different ways :

**Reading a line from a std**

Just type :
```
./run.sh std "Je suis gentil ."
```

**Reading from a file**

```
./run.sh file path_to_file
```

In that case you have an extra optional argument :
- `--out_path` : if set, the result will be written to this file.

Note that the `sh` file is simply a wrapper for the Python `main.py` file.
You can also simply use `python main.py file path_to_file` instead.

