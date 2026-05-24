import socket
from concurrent.futures import ThreadPoolExecutor

def check_port(ip, port=2087, timeout=2):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return ip
    except:
        return None

def scan_ips_from_file(input_file, output_file, threads=50):
    with open(input_file) as f:
        ips = [line.strip() for line in f if line.strip()]
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = executor.map(check_port, ips)
    working = [ip for ip in results if ip]
    with open(output_file, "w") as f:
        for ip in working:
            f.write(ip + "\n")
    print(f"Found {len(working)} open ports out of {len(ips)}. Saved to {output_file}")

if __name__ == "__main__":
    # Replace with your IP list file name
    scan_ips_from_file("my_ips.txt", "working_ips.txt")