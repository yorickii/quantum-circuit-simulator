import numpy as np
from typing import Self

class Operator():
    
    def __init__(self, factors: list[np.ndarray[complex], str]):

        arrays = []

        for factor in factors:
            if type(factor) == str:
                match factor.lower():
                    case "hadamard":
                        arrays.append(1/np.sqrt(2) * np.array( [[1, 1],[1, -1]] ))
                    case "x":
                        arrays.append(np.array( [[0, 1],[1, 0]] ))
                    case "y":
                        arrays.append(np.array( [[0, -1j],[1j, 0]] ))
                    case "z":
                        arrays.append(np.array( [[1, 0],[0, -1]] ))
                    case _:
                        raise ValueError(f"Unrecognized operator name: {factor}")
            
            elif type(factor) == np.ndarray:
                arrays.append(factor)

            else:
                raise ValueError(f"Invalid type {type(factor)} for an operator factor.")

        op = arrays[0]

        for array in arrays[1:]:
            op = np.kron(op, array)
        
        self.operator = op


    def tensor(self, argument: Self):

        op = np.kron(self.operator, argument.operator)

        return Operator([op])