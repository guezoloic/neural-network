from network import *

def data(size:int, max_val: int):
    def int_to_bits(n: int):
        return [(n >> i) & 1 
            for i in reversed(range(size))
        ]
   
    return [(int_to_bits(i),[i / max_val]) 
        for i in range(max_val + 1)
    ]

def binatodeci(binary: list[int]):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))

def train_network(network: NeuralNetwork, epochs=10000, learning_rate=0.1, 
                  verbose: bool = False, size_data: int = 8, max_val: int = 255):
    
    train_data = data(size_data, max_val)
    
    for epoch in range(epochs):
        for bits, target in train_data:
            network.backward(bits, target, learning_rate)

        if verbose and epoch % 100 == 0:
            output = network.forward(bits)[0]
            loss = (output - target[0]) ** 2

            print(f"Epoch: {epoch}, Loss: {loss:.6f} {(loss*100):.6f}%")

def main():
    size = 4
    max_val = (1 << size) - 1
    epoch_size = 6_500

    network = NeuralNetwork([size, 16, 1])

    print("Start training...")
    train_network(network, verbose=True, size_data=size, epochs=epoch_size, max_val=max_val)
    print("End training...")

    while True:
        string = input(f"Enter {size} bit number (ex: {''.join([str(random.randint(0, 1)) for i in range(size)])}) or 'quit' to close: ") \
            .strip().lower()
        
        if (string == 'quit'): break
        if (len(string) != size or any (char not in '01' for char in string)):
            print(f"Error: please enter exactly {size} bits (only 0 or 1).") 
            continue
        
        bits_input = [int(char) for char in string]
        output = network.forward(bits_input)[0] * max_val

        print("\n===== Estimated value =====")
        print(f"{output} (approx: {round(output)})")
        print("\n===== Real value =====")
        print(f"{binatodeci(bits_input)}\n")

if __name__ == "__main__":
    main()
