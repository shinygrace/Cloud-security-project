import tkinter as tk
from tkinter import filedialog, messagebox
import random
abc={
'a':0,'b':1,'c':2,'d':3,
'e':4,'f':5,'g':6,'h':7,
'i':8,'j':9,'k':10,'l':11,
'm':12,'n':13,'o':14,'p': 15,
'q':16,'r':17,'s':18,'t':19,
'u':20,'v':21,'w':22,'x':23,
'y':24,'z':25,
'A':26,'B':27,'C':28,
'D':29,'E':30,'F':31,'G': 32,
'H':33,'I':34,'J':35,'K':36,
'L':37,'M':38,'N':39,'O':40,
'P':41,'Q':42,'R':43,'S':44,
'T':45,'U':46,'V':47,'W': 48,
'X':49,'Y':50,'Z':51,',':52,
'\'':53,'.':54,'-':55,'\"':56,
' ':57, '\n':58
}

abc2={
0:'a',1:'b',2:'c',3:'d',
4:'e',5:'f',6:'g',7:'h',
8:'i',9:'j',10:'k',11:'l',
12:'m',13:'n',14:'o',15:'p',
16:'q',17:'r',18:'s',19:'t',
20:'u',21:'v',22:'w',23:'x',
24:'y',25:'z',26:'A',27:'B',
28:'C',29:'D',30:'E',31:'F',
32:'G',33:'H',34:'I',35:'J',
36:'K',37:'L',38:'M',39:'N',
40:'O',41:'P',42:'Q',43:'R',
44:'S',45:'T',46:'U',47:'V',
48:'W',49:'X',50:'Y',51:'Z',
52:',',53:'\'',54:'.',55:'-',
56:'\"',57:' ',58:'\n'
}
def generate_keys(key):
    if len(key) != 128:
        raise ValueError("Key length must be 128 bits")

    left_key = key[:64]
    right_key = key[64:]

    return left_key, right_key
Pbox={'0000':'0011','0001':'1111','0010':'1110','0011':'0000',
'0100':'0101','0101':'0100','0110':'1011','0111':'1100',
'1000':'1101','1001':'1010','1010':'1001','1011':'0110',
'1100':'0111','1101':'1000','1110':'0010','1111': '0001'}
Qbox={'0000':'1001','0001':'1110','0010':'0101','0011':'0110',
'0100':'1010','0101':'0010','0110':'0011','0111':'1100',
'1000':'1111','1001':'0000','1010':'0100','1011':'1101',
'1100':'0111','1101':'1011','1110':'0001','1111': '1000'}
def f_function_16(P):
    a1=P[0:4]
    a2=P[4:8]
    a3=P[8:12]
    a4=P[12:16]
    b1=Pbox[a1]
    b2=Qbox[a2]
    b3=Pbox[a3]
    b4=Qbox[a4] 
    a5=b1[0:2]+b2[0:2]
    a6=b1[2:4]+b3[0:2]
    a7=b2[2:4]+b4[0:2]
    a8=b3[2:4]+b4[2:4]
    b5=Qbox[a5]
    b6=Pbox[a6]
    b7=Qbox[a7]
    b8=Pbox[a8]
    a9=b5[0:2]+b6[0:2]
    a10=b5[2:4]+b7[0:2]
    a11=b6[2:4]+b8[0:2]
    a12=b7[2:4]+b8[2:4]
    b9=Pbox[a9]
    b10=Qbox[a10]
    b11=Pbox[a11]
    b12=Qbox[a12]
    f_key_output_P=b9+b10+b11+b12
    return f_key_output_P
def f_function_32(P):
    a1=P[0:4]
    a2=P[4:8]
    a3=P[8:12]
    a4=P[12:16]
    a5=P[16:20]
    a6=P[20:24]
    a7=P[24:28]
    a8=P[28:32]
    
    b1=Pbox[a1]
    b2=Qbox[a2]
    b3=Pbox[a3]
    b4=Qbox[a4]
    b5=Pbox[a5]
    b6=Qbox[a6]
    b7=Pbox[a7]
    b8=Qbox[a8] 
    
    a9=b1[0:2]+b2[0:2]
    a10=b1[2:4]+b3[0:2]
    a11=b2[2:4]+b4[0:2]
    a12=b3[2:4]+b5[0:2]
    a13=b4[2:4]+b6[0:2]
    a14=b5[2:4]+b7[0:2]
    a15=b6[2:4]+b8[0:2]
    a16=b7[2:4]+b8[2:4]
    
    b9=Qbox[a9]
    b10=Pbox[a10]
    b11=Qbox[a11]
    b12=Pbox[a12]
    b13=Qbox[a13]
    b14=Pbox[a14]
    b15=Qbox[a15]
    b16=Pbox[a16]
    
    a17=b9[0:2]+b10[0:2]
    a18=b9[2:4]+b11[0:2]
    a19=b10[2:4]+b12[0:2]
    a20=b11[2:4]+b13[0:2]
    a21=b12[2:4]+b14[0:2]
    a22=b13[2:4]+b15[0:2]
    a23=b14[2:4]+b16[0:2]
    a24=b15[2:4]+b16[2:4]
    
    b17=Pbox[a17]
    b18=Qbox[a18]
    b19=Pbox[a19]
    b20=Qbox[a20]
    b21=Pbox[a21]
    b22=Qbox[a22]
    b23=Pbox[a23]
    b24=Qbox[a24]
    f_key_output_P=b17+b18+b19+b20+b21+b22+b23+b24
    return f_key_output_P
def xor_func(bin1,bin2):
    xor_result = int(bin1, 2) ^ int(bin2, 2)
    xor_result_final = bin(xor_result)[2:].zfill(16)
    return xor_result_final
def xor_func_32(bin1, bin2):
    xor_result = int(bin1, 2) ^ int(bin2, 2)
    xor_result_final = bin(xor_result)[2:].zfill(32)
    return xor_result_final
def binary_add(bin1,bin2):
    num1 = int(bin1, 2)
    num2 = int(bin2, 2)
    result = num1 + num2
    K = bin(result)[2:].zfill(16)
    return K
def left_circular_shift_16bit(binary_string):
    if len(binary_string) != 16:
        raise ValueError("Input string length must be 16")
    return binary_string[16:] + binary_string[:16]
def left_regular_shift_2bit(binary_string):
    if len(binary_string) != 16:
        raise ValueError("Input string length must be 16")
    return binary_string[1:] + binary_string[0]
def xnor_func(bin1, bin2):
    xnor_result = int(bin1, 2) ^ int(bin2, 2) ^ 0xFFFF
    xnor_result_final = bin(xnor_result)[2:].zfill(16)
    return xnor_result_final
def xnor_func_32(bin1, bin2):
    xnor_result = int(bin1, 2) ^ int(bin2, 2) ^ 0xFFFFFFFF
    xnor_result_final = bin(xnor_result)[2:].zfill(32)
    return xnor_result_final
def xnor_func_64(bin1, bin2):
    xnor_result = int(bin1, 2) ^ int(bin2, 2) ^ 0xFFFFFFFF
    xnor_result_final = bin(xnor_result)[2:].zfill(64)
    return xnor_result_final
def right_key_op(right_key):
    PR1=right_key[0:16]
    PR2=right_key[16:32]
    PR3=right_key[32:48]
    PR4=right_key[48:64]
    shifted_PR1 = left_circular_shift_16bit(PR1)
    shifted_PR2 = left_circular_shift_16bit(PR2)
    shifted_PR3 = left_circular_shift_16bit(PR3)
    shifted_PR4 = left_circular_shift_16bit(PR4)
    untransposed_right_matrix = [[0 for _ in range(4)] for _ in range(4)]
    a=0
    pr1=list(PR1)
    for i in range(4):
        for j in range(4):
            untransposed_right_matrix[i][j]=pr1[a]
            a=a+1
            
    transposed_right_matrix = [[row[i] for row in untransposed_right_matrix] for i in range(4)]
    PR2_f_func_op=f_function_16(PR2)
    xor=xor_func(shifted_PR1,PR2_f_func_op)
    K1=xor+PR2_f_func_op
    PR3_f_func_op=f_function_16(PR3)
    double_shifted_PR4=left_regular_shift_2bit(shifted_PR4)
    K2=PR3_f_func_op+double_shifted_PR4
    KK=xnor_func(K1,K2)
    return K1,K2,KK
def left_key_op(left_key):
    PL1=left_key[0:16]
    PL2=left_key[16:32]
    PL3=left_key[32:48]
    PL4=left_key[48:64]
    shifted_PL1 = left_circular_shift_16bit(PL1)
    shifted_PL2 = left_circular_shift_16bit(PL2)
    shifted_PL3 = left_circular_shift_16bit(PL3)
    shifted_PL4 = left_circular_shift_16bit(PL4)
    untransposed_left_matrix = [[0 for _ in range(4)] for _ in range(4)]
    a=0
    pl3=list(PL3)
    for i in range(4):
        for j in range(4):
            untransposed_left_matrix[i][j]=pl3[a]
            a=a+1
    transposed_left_matrix = [[row[i] for row in untransposed_left_matrix] for i in range(4)]
    PL2_f_func_op=f_function_16(PL2)
    xor=xor_func(shifted_PL1,PL2_f_func_op)
    K4=xor+PL2_f_func_op
    PL1_f_func_op=f_function_16(PL1)
    double_shifted_PL2=left_regular_shift_2bit(shifted_PL2)
    xor=xor_func(PL1_f_func_op,double_shifted_PL2)
    K3=xor+double_shifted_PL2
    PL4_f_func_op=f_function_16(PL4)
    K4=shifted_PL3+PL4_f_func_op
    KK1=xnor_func(K3,K4)
    return K3,K4,KK1
def layer1_encrypt(text, key):
    plaintext=text
    choice=key
    key=""
    for i in choice:
        j=abc[i]
        k=hex(j).lstrip("0x").rstrip("L")
        res = "{0:08b}".format(int(k, 16)) 
        key=key+res        
    left_key, right_key=generate_keys(key)
    K1,K2,KK=right_key_op(right_key)
    K3,K4,KK1=left_key_op(left_key)
    SK=KK+KK1
    M1 = plaintext[:32]
    M2 = plaintext[32:64]
    M3 = plaintext[64:96]
    M4 = plaintext[96:]
    R1_M1 = xor_func_32(M1,K1)
    R1_M2 = xor_func_32(R1_M1,M2)
    R1_M3 = xor_func_32(R1_M2, M3)
    R1_F_output = f_function_32(R1_M3)
    R1_M4 = xnor_func_32(R1_F_output, M4)
    R2_M1,R2_M2=R1_M2,R1_M1
    R2_M3,R2_M4=R1_M4,R1_M3
    R3_M4 = xor_func_32(R2_M4,K2)
    R3_M3 = xor_func_32(R3_M4,R2_M3)
    R2_F_output = f_function_32(R3_M3)
    R3_M2 = xor_func_32(R2_F_output,R2_M2)
    R3_M1 = xnor_func_32(R3_M2,R2_M1)
    S_R3_M1= xor_func_32(R3_M1,K3)
    R3_F_output = f_function_32(S_R3_M1)
    S_R3_M2= xor_func_32(R3_F_output,R3_M2)
    S_R3_M3= xor_func_32(S_R3_M2,R3_M3)
    S_R3_M4= xnor_func_32(S_R3_M3,R3_M4)
    R4_M1,R4_M2=S_R3_M2,S_R3_M1
    R4_M3,R4_M4=S_R3_M4,S_R3_M3
    R5_M4 = xor_func_32(R4_M4,K4)
    R5_M3 = xor_func_32(R5_M4,R4_M3)
    R4_F_output = f_function_32(R5_M3)
    R5_M2 = xor_func_32(R4_F_output,R4_M2)
    R5_M1 = xnor_func_32(R5_M2,R4_M1)
    S_R5_M1= xor_func_32(R5_M1,KK)
    S_R5_M2= xor_func_32(S_R5_M1,R5_M2)
    S_R5_M3= xor_func_32(S_R5_M2,R5_M3)
    R5_F_output = f_function_32(S_R5_M3)
    S_R5_M4= xnor_func_32(R5_F_output,R5_M4)
    R6_M1,R6_M2=S_R5_M2,S_R5_M1
    R6_M3,R6_M4=S_R5_M4,S_R5_M4
    I_R6_M4 = xor_func_32(R6_M4,KK1)
    I_R6_M3 = xor_func_32(I_R6_M4,R6_M3)
    R6_F_output = f_function_32(I_R6_M3)
    I_R6_M2 = xor_func_32(R6_F_output,R6_M2)
    I_R6_M1= xnor_func_32(I_R6_M2,R6_M1)
    MP1= I_R6_M1+I_R6_M2
    MP2= I_R6_M3+I_R6_M4
    C_MP1=xnor_func_64(MP1,SK)
    C_MP2=xnor_func_64(C_MP1,MP2)
    ciphertext=C_MP1+C_MP2
    return ciphertext
def layer2_encrypt(text, key):
    text1 = input_text.get("1.0", "end-1c")
    key1 = key_entry.get()
    p_text=text1
    blocks = [p_text[i:i+16] for i in range(0, len(p_text), 16)]
    c_txt=""
    e_txt=""
    for p in blocks:
        txt=""
        for ch in  p:
                j=abc[ch]
                if j==0:
                    k=format(j, '02x') 
                    res=format(j, '08b')
                else:
                    k=hex(j).lstrip("0x").rstrip("L")
                    res = "{0:08b}".format(int(k, 16)) 
                txt=txt+res
        e_txt +=txt
        enc=layer1_encrypt(txt,key)
        c_txt +=enc
    ce = c_txt
    cipher_rsa=""
    binarye=""
    for i in range(0,len(ce), 8):
        binary_block = ce[i:i+8]
        if binary_block=="00000000":
            hex_block='00'
        groups_of_four = [binary_block[i:i+4] for i in range(0, len(binary_block), 4)]
        hexd = ''.join(hex(int(group, 2))[2:].upper() for group in groups_of_four)
        decimal_value = int(hexd, 16)
        if(decimal_value>58):
            decimal_value=decimal_value % 59
        else:
            decimal_value=decimal_value
        binary = format(decimal_value, '08b') 
        cipher_rsa+=abc2[decimal_value]
        binarye+=binary
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a
    def mod_inverse(a, m):
        m0, x0, x1 = m, 0, 1
        while a > 1:
            q = a // m
            m, a = a % m, m
            x0, x1 = x1 - q * x0, x0
        return x1 + m0 if x1 < 0 else x1
    def generate_keypair(p, q):
        n = p * q
        phi = (p - 1) * (q - 1)
        e = random.randrange(1, phi)
        while gcd(e, phi) != 1:
            e = random.randrange(1, phi)
        d = mod_inverse(e, phi)
        return ((e, n), (d, n))
    def encrypt(pk, plaintext):
        key, n = pk
        cipher = [(ord(char) ** key) % n for char in plaintext]
        return cipher
    if __name__ == '__main__':
        p = 61
        q = 53
        public, private = generate_keypair(p, q)
        message = cipher_rsa
        encrypted_msg = encrypt(public, message)
        decrypted_msg = layer1_decrypt(private, encrypted_msg)
    return encrypted_msg ,decrypted_msg, e_txt
def layer1_decrypt(pk, ciphertext_rsa):
    key, n = pk
    plain = [chr((char ** key) % n) for char in ciphertext_rsa]
    return ''.join(plain)
def layer2_decrypt(de, key):
    choice=key
    key=""
    for i in choice:
        j=ord(str(i))
        k=hex(j).lstrip("0x").rstrip("L")
        res = "{0:08b}".format(int(k, 16)) 
        key=key+res
    left_key, right_key=generate_keys(key)
    K1,K2,KK=right_key_op(right_key)
    K3,K4,KK1=left_key_op(left_key)
    SK=KK+KK1
    D1_MP1=de[0:64]
    D1_MP2=de[64:128]
    D2_MP1=xnor_func_64(D1_MP1,SK)
    D2_MP2=xnor_func_64(D2_MP1,D1_MP2)
    R6_M1=D2_MP1[0:32]
    R6_M2=D2_MP1[32:64]
    R6_M3=D2_MP2[0:32]
    R6_M4=D2_MP2[32:64]
    S_R6_M4=xor_func_32(R6_M4,KK1)
    S_R6_M3=xor_func_32(R6_M3,S_R6_M4)
    R6_F_OUTPUT=f_function_32(S_R6_M3)
    S_R6_M2=xor_func_32(R6_M2,R6_F_OUTPUT)
    S_R6_M1=xnor_func_32(S_R6_M2,R6_M1)
    R5_M1,R5_M2=S_R6_M2,S_R6_M1
    R5_M3,R5_M4=S_R6_M4,S_R6_M3
    I_R5_M1= xor_func_32(KK,R5_M1)
    R5_F_OUTPUT=f_function_32(I_R5_M1)
    I_R5_M2= xor_func_32(R5_F_OUTPUT,R5_M2)
    I_R5_M3= xor_func_32(I_R5_M2,R5_M3)
    I_R5_M4=xnor_func_32(I_R5_M3,R5_M4)
    R4_M4=xor_func_32(K4,I_R5_M4)
    R4_M3=xor_func_32(I_R5_M3,R4_M4)
    R4_F_OUTPUT=f_function_32(R4_M3)
    R4_M2=xor_func_32(I_R5_M2, R4_F_OUTPUT)
    R4_M1=xnor_func_32(R4_M2,I_R5_M1)
    R3_M1,R3_M2=R4_M2,R4_M1
    R3_M3,R3_M4=R4_M4,R4_M3
    R2_M1=xnor_func_32(K3,R3_M1)
    R2_M2=xor_func_32(R3_M2,R2_M1)
    R2_M3=xor_func_32(R3_M3,R2_M2)
    R3_F_OUTPUT=f_function_32(R2_M3)
    R2_M4=xnor_func_32(R3_M4,R3_F_OUTPUT)
    S_R2_M4=xor_func_32(K2,R2_M4)
    S_R2_M3=xor_func_32(R2_M3, S_R2_M4)
    R2_F_OUTPUT=f_function_32(S_R2_M3)
    S_R2_M2=xor_func_32(R2_M2,R2_F_OUTPUT)
    S_R2_M1=xnor_func_32(R2_M1,S_R2_M2)
    R1_M1,R1_M2=S_R2_M2,S_R2_M1
    R1_M3,R1_M4=S_R2_M4,S_R2_M3
    I_R1_M1= xor_func_32(K1,R1_M1)
    R1_F_OUTPUT=f_function_32(I_R1_M1)
    I_R1_M2= xor_func_32(R1_F_OUTPUT,R1_M2)
    I_R1_M3= xor_func_32(I_R1_M2,R1_M3)
    I_R1_M4=xnor_func_32(I_R1_M3,R1_M4)
    plaintxt_de=I_R1_M1+I_R1_M2+I_R1_M3+I_R1_M4
    ptxt=""
    for i in range(0, 128, 8):
        binary_block = plaintxt_de[i:i+8]
        try:
            hex_block = hex(int(binary_block, 2)).lstrip("0x")
            decimal_value = int(hex_block, 16)
            while decimal_value > 128:
                decimal_value =decimal_value % 128
            ptxt += chr(decimal_value)
        except ValueError:
            pass
    return ptxt
def browse_file():
    filename = filedialog.askopenfilename()
    if filename:
        with open(filename, "r") as file:
            text = file.read()
            input_text.delete("1.0", "end")
            input_text.insert("1.0", text)
def layer1_encrypt_text():
    text = input_text.get("1.0", "end-1c")
    key = key_entry.get()
    if len(key) != 16:
        messagebox.showerror("Error", "Key must be 16 characters long")
        return
    p_text=text
    blocks = [p_text[i:i+16] for i in range(0, len(p_text), 16)]
    c_txt=""
    e_txt=""
    for p in blocks:
        txt=""
        for ch in  p:
                j=abc[ch]
                if j==0:
                    k=format(j, '02x') 
                    res=format(j, '08b')
                else:
                    k=hex(j).lstrip("0x").rstrip("L")
                    res = "{0:08b}".format(int(k, 16)) 
                txt=txt+res
        e_txt +=txt
        enc=layer1_encrypt(txt,key)
        c_txt +=enc
    ce=c_txt
    cipher_rsa=""
    binarye=""
    for i in range(0,len(ce), 8):
        binary_block = ce[i:i+8]
        if binary_block=="00000000":
            hex_block='00'
        groups_of_four = [binary_block[i:i+4] for i in range(0, len(binary_block), 4)]
        hexd = ''.join(hex(int(group, 2))[2:].upper() for group in groups_of_four)
        decimal_value = int(hexd, 16)
        if(decimal_value>58):
            decimal_value=decimal_value % 59
        else:
            decimal_value=decimal_value
        binary = format(decimal_value, '08b')
        cipher_rsa+=abc2[decimal_value]
        binarye+=binary
    intermediate_encrypted_text = cipher_rsa
    intermediate_encrypted_output_text.delete("1.0", "end")
    intermediate_encrypted_output_text.insert("1.0", intermediate_encrypted_text)
def layer2_encrypt_text():
    text = intermediate_encrypted_output_text.get("1.0", "end-1c")
    key = key_entry.get()   
    if len(key) != 16:
        messagebox.showerror("Error", "Key must be 16 characters long")
        return
    encrypted_cipher_text, decrypted_text, ee = layer2_encrypt(text, key)
    encrypted_output_text.delete("1.0", "end")
    encrypted_output_text.insert("1.0", encrypted_cipher_text)
def layer1_decrypt_text():
    text1 = intermediate_encrypted_output_text.get("1.0", "end-1c")
    key1 = key_entry.get()
    if len(key1) != 16:
        messagebox.showerror("Error", "Key must be 16 characters long")
        return
    cipher_rsa = encrypted_output_text.get("1.0", "end-1c")
    text3 = intermediate_encrypted_output_text.get("1.0", "end-1c")
    key3 = key_entry.get()   
    if len(key3) != 16:
        messagebox.showerror("Error", "Key must be 16 characters long")
        return
    encrypted_cipher_text, decrypted_text, ee = layer2_encrypt(text3, key3)
    intermediate_decrypted_text = decrypted_text
    intermediate_decrypted_output_text.delete("1.0", "end")
    intermediate_decrypted_output_text.insert("1.0", intermediate_decrypted_text)
def layer2_decrypt_text():
    text = intermediate_encrypted_output_text.get("1.0", "end-1c")
    key = key_entry.get()   
    if len(key) != 16:
        messagebox.showerror("Error", "Key must be 16 characters long")
        return
    encrypted_cipher_text, decrypted_text, ee = layer2_encrypt(text, key)
    plaintxt_de1 = ee
    result_hex = []
    for i in range(0, len(plaintxt_de1), 8):
        block = plaintxt_de1[i:i+8]
        decimal_value = int(block, 2)
        hex_value = hex(decimal_value)[2:].upper()
        if len(hex_value) < 2:
            hex_value = "0" + hex_value
        result_hex.append(hex_value)


    result_chars = ''
    for hex_string in result_hex:
        decimal_value = int(hex_string, 16)
        char = abc2.get(decimal_value, '')
        result_chars += char
    decrypted_plain_text = result_chars
    decrypted_output_text.delete("1.0", "end")
    decrypted_output_text.insert("1.0", decrypted_plain_text)
root = tk.Tk()
root.title("Layered Encryption & Decryption")
browse_button = tk.Button(root, text="Browse Files", command=browse_file)
browse_button.pack()
label = tk.Label(root, text="Enter Text:")
label.pack()
input_text = tk.Text(root, height=5, width=50)
input_text.pack()
key_label = tk.Label(root, text="Enter Key (16 characters):")
key_label.pack()
key_entry = tk.Entry(root)
key_entry.pack()
layer1_encrypt_button = tk.Button(root, text="Layer 1 Encrypt", command=layer1_encrypt_text)
layer1_encrypt_button.pack()
intermediate_encrypted_label = tk.Label(root, text="Intermediate Encrypted Text:")
intermediate_encrypted_label.pack()
intermediate_encrypted_output_text = tk.Text(root, height=5, width=50)
intermediate_encrypted_output_text.pack()
layer2_encrypt_button = tk.Button(root, text="Layer 2 Encrypt", command=layer2_encrypt_text)
layer2_encrypt_button.pack()
encrypted_label = tk.Label(root, text="Encrypted Cipher Text:")
encrypted_label.pack()
encrypted_output_text = tk.Text(root, height=5, width=50)
encrypted_output_text.pack()
layer1_decrypt_button = tk.Button(root, text="Layer 1 Decrypt", command=layer1_decrypt_text)
layer1_decrypt_button.pack()
intermediate_decrypted_label = tk.Label(root, text="Intermediate Decrypted Text:")
intermediate_decrypted_label.pack()
intermediate_decrypted_output_text = tk.Text(root, height=5, width=50)
intermediate_decrypted_output_text.pack()
layer2_decrypt_button = tk.Button(root, text="Layer 2 Decrypt", command=layer2_decrypt_text)
layer2_decrypt_button.pack()
decrypted_label = tk.Label(root, text="Decrypted Plain Text:")
decrypted_label.pack()
decrypted_output_text = tk.Text(root, height=5, width=50)
decrypted_output_text.pack()
root.mainloop()
