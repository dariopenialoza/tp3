import json

def main3b():
    print('EJERCICIO 3 B')
    with open('./config3b.json', 'r') as f:
        configData = json.load(f)
        f.close()
        
    
if __name__ == "__main__":
    main3b()