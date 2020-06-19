from enum import Enum

'''
0 – standardowe, puste pole, po którym agent moze sie poruszac
1 – sciana, która nie moze zostac pokonana (Uwaga: sciany beda znajdowac
sie jedynie na obrzezach, nie bedzie scian wewnatrz labiryntu).
2 - tunel pionowy
3 - tunel poziomy
5 – symbol agenta, oznaczajacy jego pozycje poczatkowa (Uwaga: nie ma
koniecznosci wizualizacji kolejnych kroków).
8 – symbol wyjscia, oznaczajacy pozycje celu, na który agent powinien
dotrzec (Uwaga: jest dokładnie jeden symbol 8 oraz znajduje sie on na
obrzezu).
'''


class Field(Enum):
    EMPTY = 0
    WALL = 1
    VERTICAL_TUNNEL = 2
    HORIZONTAL_TUNNEL = 3
    AGENT = 5
    EXIT = 8
