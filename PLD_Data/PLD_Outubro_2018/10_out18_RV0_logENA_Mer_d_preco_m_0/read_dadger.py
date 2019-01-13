# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:34:41 2018

@author: felip
"""

import re


f = open("Dadger.RV1","r") #opens file with name of "test.txt"
dadger = f.read()
f.close()

dadgerFormatado = ""

linha = 0
for line in dadger.splitlines():
    if not line.startswith('&'):
        line = re.sub(r"\s+", ",", line)
        dadgerFormatado = dadgerFormatado + line + "\n"
        
f2 = open("DadgerFormatado.csv", "w")
f2.write(dadgerFormatado)
f2.close()