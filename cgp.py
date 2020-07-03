import cgpy as cgp

op_table = [Operation("+"), Operation("*"), Operation("sin")]

dims = 2
nr_of_parameters = 0
nr_of_nodes = 5
cgp = create_random_cgp(dims, nr_of_parameters, op_table, nr_of_nodes)

pnt = [0.5, 1.5]
print(cgp.eval(pnt))