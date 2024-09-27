import hashlib
import os
from binascii import unhexlify, hexlify
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from tqdm import tqdm
import time

# Função para gerar um endereço Bitcoin a partir de uma chave privada
def gerar_endereco_bitcoin(chave_privada_hex):
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

# Função para codificar em base58
def encode_base58(b):
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

# Função para buscar chaves em um intervalo utilizando a CPU
def busca_cpu(inicio, fim):
    total = fim - inicio
    keys_processed = 0
    start_time = time.time()

    with tqdm(total=total, desc='Buscando na CPU', unit='chave') as pbar:
        for chave_privada in range(inicio, fim):
            chave_privada_hex = hex(chave_privada)[2:]  # Remove o '0x'
            if len(chave_privada_hex) < 64:
                chave_privada_hex = chave_privada_hex.zfill(64)  # Preenche com zeros à esquerda
            endereco = gerar_endereco_bitcoin(chave_privada_hex)
            # Aqui você pode adicionar a lógica para verificar o saldo ou outras condições
            # Por exemplo:
            # saldo = verificar_saldo(endereco)
            # if saldo > 0:
            #     salvar_chave(chave_privada_hex, endereco, saldo)

            keys_processed += 1
            pbar.update(1)

            # Calcular e exibir chaves por segundo a cada 1000 chaves
            if keys_processed % 1000 == 0:
                elapsed_time = time.time() - start_time
                cps = keys_processed / elapsed_time
                print(f'Chaves por segundo (CPU): {cps:.2f}')
                start_time = time.time()
                keys_processed = 0

# Função para buscar chaves em um intervalo utilizando a GPU
def busca_gpu(inicio, fim):
    # Definir o módulo CUDA
    mod = SourceModule("""
    __global__ void busca_chaves(unsigned long long *chaves, int *resultados) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        // Exemplo: apenas preencher com a chave
        if (idx < 1024) {
            resultados[idx] = chaves[idx];  // Simulando a busca
        }
    }
    """)

    # Definir o kernel
    func = mod.get_function("busca_chaves")

    total = fim - inicio
    batch_size = 1024  # Número de chaves processadas por vez na GPU
    keys_processed = 0
    start_time = time.time()

    with tqdm(total=total, desc='Buscando na GPU', unit='chave') as pbar:
        for batch_start in range(inicio, fim, batch_size):
            batch_end = min(batch_start + batch_size, fim)
            current_batch_size = batch_end - batch_start

            # Preparar os dados para a GPU
            chaves = np.array(range(batch_start, batch_end), dtype=np.uint64)
            resultados = np.zeros(current_batch_size, dtype=np.int32)

            # Alocar memória na GPU
            chaves_gpu = cuda.mem_alloc(chaves.nbytes)
            resultados_gpu = cuda.mem_alloc(resultados.nbytes)

            # Copiar dados para a GPU
            cuda.memcpy_htod(chaves_gpu, chaves)

            # Executar o kernel
            func(chaves_gpu, resultados_gpu, block=(256, 1, 1), grid=(int(np.ceil(current_batch_size / 256)), 1))

            # Copiar resultados de volta para a CPU
            cuda.memcpy_dtoh(resultados, resultados_gpu)

            # Processar os resultados (aqui apenas simulando)
            # Você pode adicionar a lógica para verificar saldo ou outras condições

            keys_processed += current_batch_size
            pbar.update(current_batch_size)

            # Calcular e exibir chaves por segundo a cada 1024 chaves
            if keys_processed >= batch_size:
                elapsed_time = time.time() - start_time
                cps = keys_processed / elapsed_time
                print(f'Chaves por segundo (GPU): {cps:.2f}')
                start_time = time.time()
                keys_processed = 0

# Carregar intervalos de chaves a partir do arquivo
def carregar_intervalos(arquivo):
    intervalos = []
    with open(arquivo, "r") as f:
        for linha in f:
            inicio, fim = linha.strip().split(":")
            inicio = int(inicio, 16)  # Converte para inteiro
            fim = int(fim, 16)        # Converte para inteiro
            intervalos.append((inicio, fim))
    return intervalos

if __name__ == "__main__":
    intervalos = carregar_intervalos("intervalos.txt")

    # Processar cada intervalo
    for intervalo in intervalos:
        inicio, fim = intervalo
        print(f'\nProcessando intervalo: {hex(inicio)} a {hex(fim)}')

        # Executar busca em CPU
        busca_cpu(inicio, fim)

        # Executar busca em GPU
        busca_gpu(inicio, fim)

    print("Busca concluída.")
