import hashlib
from binascii import unhexlify, hexlify
import numpy as np
from tqdm import tqdm
import time
import logging
import os

# Configuração do Logger
logging.basicConfig(
    filename='program.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Função para gerar um endereço Bitcoin a partir de uma chave privada
def gerar_endereco_bitcoin(chave_privada_hex):
    try:
        # Converter a chave privada de hexadecimal para decimal
        chave_privada = int(chave_privada_hex, 16)

        # Aplicar SHA256 na chave privada
        sha256 = hashlib.sha256(unhexlify(chave_privada_hex)).digest()
        # Aplicar RIPEMD160 na saída do SHA256
        ripemd160 = hashlib.new('ripemd160', sha256).digest()

        # Adicionar versão (0x00 para Bitcoin Mainnet)
        versioned_payload = b'\x00' + ripemd160

        # Calcular o checksum
        checksum = hashlib.sha256(hashlib.sha256(versioned_payload).digest()).digest()[:4]

        # Concatenar payload e checksum
        final_payload = versioned_payload + checksum

        # Converter para base58
        endereco_bitcoin = encode_base58(final_payload)
        return endereco_bitcoin
    except Exception as e:
        logging.error(f"Erro ao gerar endereço Bitcoin: {e}")
        return None

# Função para codificar em base58
def encode_base58(b):
    try:
        alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        num = int.from_bytes(b, 'big')
        arr = []
        while num > 0:
            num, rem = divmod(num, 58)
            arr.append(alphabet[rem])
        # Tratar os zeros à esquerda
        for byte in b:
            if byte == 0:
                arr.append(alphabet[0])
            else:
                break
        return b''.join(arr[::-1]).decode('utf-8')
    except Exception as e:
        logging.error(f"Erro ao codificar base58: {e}")
        return None

# Função para carregar endereços Bitcoin do arquivo
def carregar_enderecos(arquivo):
    enderecos = set()
    try:
        with open(arquivo, "r") as f:
            for linha in f:
                partes = linha.strip().split()
                if len(partes) >= 2:
                    endereco = partes[1]
                    enderecos.add(endereco)
        logging.info(f"{len(enderecos)} endereços carregados de {arquivo}.")
    except FileNotFoundError:
        logging.error(f"Arquivo {arquivo} não encontrado.")
    except Exception as e:
        logging.error(f"Erro ao carregar endereços: {e}")
    return enderecos

# Função para salvar chaves privadas encontradas
def salvar_chave(chave_privada_hex, endereco, arquivo='chaves_encontradas.txt'):
    try:
        with open(arquivo, 'a') as f:
            f.write(f'{chave_privada_hex} {endereco}\n')
        logging.info(f"Chave encontrada e salva: {chave_privada_hex} {endereco}")
    except Exception as e:
        logging.error(f"Erro ao salvar chave: {e}")

# Função para salvar a última chave testada
def salvar_ultima_chave_testada(inicio, fim, search_type, last_key, arquivo='ultimas_chaves_testadas.txt'):
    try:
        entry = f'{hex(inicio)}:{hex(fim)}:{search_type}:{hex(last_key)}\n'
        with open(arquivo, 'a') as f:
            f.write(entry)
        logging.info(f"Última chave testada salva para intervalo {hex(inicio)}:{hex(fim)} - {search_type}: {hex(last_key)}")
    except Exception as e:
        logging.error(f"Erro ao salvar a última chave testada: {e}")

# Função para buscar chaves em um intervalo utilizando a CPU
def busca_cpu(inicio, fim, enderecos):
    total = fim - inicio
    keys_processed = 0
    start_time = time.time()
    last_key_testada = inicio

    with tqdm(total=total, desc='Buscando na CPU', unit='chave') as pbar:
        for chave_privada in range(inicio, fim):
            chave_privada_hex = hex(chave_privada)[2:]  # Remove o '0x'
            if len(chave_privada_hex) < 64:
                chave_privada_hex = chave_privada_hex.zfill(64)  # Preenche com zeros à esquerda
            endereco = gerar_endereco_bitcoin(chave_privada_hex)

            # Verificar se o endereço está no conjunto de endereços
            if endereco and endereco in enderecos:
                print(f'-> Encontrado! Chave Privada: {chave_privada_hex}, Endereço: {endereco}')
                salvar_chave(chave_privada_hex, endereco)

            keys_processed += 1
            pbar.update(1)
            last_key_testada = chave_privada

            # Calcular e exibir chaves por segundo a cada 1000 chaves
            if keys_processed >= 1000:
                elapsed_time = time.time() - start_time
                cps = keys_processed / elapsed_time if elapsed_time > 0 else 0
                logging.info(f'Chaves por segundo (CPU): {cps:.2f}')
                print(f'Chaves por segundo (CPU): {cps:.2f}')
                start_time = time.time()
                keys_processed = 0

    # Salvar a última chave testada após o término da busca
    salvar_ultima_chave_testada(inicio, fim, 'CPU', last_key_testada)

# Função para buscar chaves em um intervalo utilizando a CPU (Substitui GPU)
# Como estamos removendo a GPU, todas as buscas serão feitas na CPU
# Então, podemos remover a função busca_gpu completamente ou mantê-la para uniformidade
# Neste caso, vamos removê-la para evitar confusão

# Função para carregar intervalos de chaves a partir do arquivo
def carregar_intervalos(arquivo):
    intervalos = []
    try:
        with open(arquivo, "r") as f:
            for linha in f:
                inicio, fim = linha.strip().split(":")
                inicio = int(inicio, 16)  # Converte para inteiro
                fim = int(fim, 16)        # Converte para inteiro
                intervalos.append((inicio, fim))
        logging.info(f"{len(intervalos)} intervalos carregados de {arquivo}.")
    except FileNotFoundError:
        logging.error(f"Arquivo {arquivo} não encontrado.")
    except Exception as e:
        logging.error(f"Erro ao carregar intervalos: {e}")
    return intervalos

def main():
    logging.info("Programa iniciado.")

    # Carregar endereços
    enderecos = carregar_enderecos("bitcoin_address.txt")
    print(f'Total de endereços carregados: {len(enderecos)}')

    # Carregar intervalos
    intervalos = carregar_intervalos("intervalos.txt")

    # Processar cada intervalo
    for intervalo in intervalos:
        inicio, fim = intervalo
        print(f'\nProcessando intervalo: {hex(inicio)} a {hex(fim)}')
        logging.info(f'Iniciando busca no intervalo {hex(inicio)} a {hex(fim)}')

        # Executar busca em CPU
        busca_cpu(inicio, fim, enderecos)

        logging.info(f'Busca concluída no intervalo {hex(inicio)} a {hex(fim)}')

    print("Busca concluída.")
    logging.info("Programa finalizado.")

if __name__ == "__main__":
    main()
