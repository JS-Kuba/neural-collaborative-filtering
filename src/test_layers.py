num_layers=3
factor_num = 16
for i in range(num_layers):
    input_size = factor_num * (2 ** (num_layers - i))
    print(input_size, input_size//2)

print(40*"-")
layers = [128, 64, 32, 16]
for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
    print(in_size, out_size)