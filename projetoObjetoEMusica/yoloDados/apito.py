import winsound
import time

def emitir_apito(freq, duracao):
    winsound.Beep(freq, duracao)

# Emitir três apitos de 3 segundos cada
freq = 1000  # Frequência em Hertz
duracao_apito = 1000  # Duração de cada apito em milissegundos (3 segundos)

for _ in range(3):
    emitir_apito(freq, duracao_apito)
    time.sleep(1)  # Pausa de 1 segundo entre os apitos
