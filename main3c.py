import json

def main3a():
    print('EJERCICIO 3 C')
    with open('./config3.json', 'r') as f:
        configData = json.load(f)
        f.close()
        
    
if __name__ == "__main__":
    main3a()