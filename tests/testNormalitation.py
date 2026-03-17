from NLP.normalizacion import TextNormalizer

normalizar = TextNormalizer()

text = "Heeeyyy @john!!! I'm soooo HAPPY right now 😂😂 #excited but kinda tired toooo... " \
"lol, you know how life is. Yesterday I went 2 the moviesss and it was 'amazingggg'!!! BTW, " \
"the soundtrack was 🔥🔥🔥. Not sure if I'll go tomorrow, cuz I'm super busy w/ workkkk."

normalizado = normalizar.clean_text(text)
print(normalizado)

"""
Observaciones:
Añadir una función que elimine los caracteres repetidos como:
soooo -> soo
toooo -> too
moviesss -> movies
amazingggg -> amazing
"""