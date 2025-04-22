#!/usr/bin/env python3
import os
import time
import argparse
import asyncio
import multiprocessing
import aiohttp
import aiofiles
import requests
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# Silence SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_DATA_DIR = "./data/images"
DEFAULT_TIMEOUT  = 20    # ?
DEFAULT_CONC     = 10    # ?
DEFAULT_THREADS  = 10    # ?
BASE_URL         = "https://digi.ub.uni-heidelberg.de/diglitData/image/cpl1969/4/{page:03d}{side}.jpg"

# URL-Generierung
def generate_image_urls(start: int, end: int):
    return [BASE_URL.format(page=p, side=side) for p in range(start, end+1) for side in ('r','v')]

# Setup
def ensure_data_dir(path: str):
    os.makedirs(path, exist_ok=True)


async def download_image_async(session, url, data_dir):
    start = time.time()
    async with session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
        resp.raise_for_status()
        data = await resp.read()
    fname = os.path.join(data_dir, os.path.basename(url))
    async with aiofiles.open(fname, 'wb') as f:
        await f.write(data)
    duration = time.time() - start
    print(f"[Hybrid] Downloaded {os.path.basename(url)} in {duration:.2f}s")
    return duration, len(data)

async def download_chunk_async(urls, data_dir, concurrency):
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image_async(session, url, data_dir) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    cleaned = [(t,s) for res in results if isinstance(res, tuple) for t,s in [res]]
    return cleaned

def hybrid_worker(chunk_urls, data_dir, concurrency, idx, total):
    print(f"[Hybrid] Process {idx}/{total} starting {len(chunk_urls)} downloads...")
    results = asyncio.run(download_chunk_async(chunk_urls, data_dir, concurrency))
    print(f"[Hybrid] Process {idx}/{total} completed")
    return results

def run_hybrid(urls, num_procs, concurrency, data_dir):
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir): os.remove(os.path.join(data_dir, f))
    else:
        ensure_data_dir(data_dir)
    chunks = [urls[i::num_procs] for i in range(num_procs)]
    args_list = [(chunks[i], data_dir, concurrency, i+1, num_procs) for i in range(num_procs)]
    wall_start = time.time()
    with multiprocessing.Pool(processes=num_procs) as pool:
        proc_results = pool.starmap(hybrid_worker, args_list)
    wall_time = time.time() - wall_start
    times, sizes = [], []
    for proc in proc_results:
        for t, s in proc:
            times.append(t); sizes.append(s)
    sum_time = sum(times)
    total_mb = sum(sizes) / (1024*1024)
    throughput_wall = total_mb / wall_time if wall_time > 0 else 0
    print(f"[Hybrid] All processes done: wall {wall_time:.2f}s, sum durations {sum_time:.2f}s, throughput {throughput_wall:.2f} MB/s")
    return {'mode': f'hybrid_{num_procs}proc', 'wall_s': wall_time, 'sum_s': sum_time, 'mb_per_s': throughput_wall}

def run_sequential(urls, data_dir, timeout):
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir): os.remove(os.path.join(data_dir, f))
    else:
        ensure_data_dir(data_dir)
    session = requests.Session()
    session.verify = False
    wall_start = time.time()
    total_sizes, times = 0, []
    for idx, url in enumerate(urls, 1):
        start = time.time()
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.content
        fname = os.path.join(data_dir, os.path.basename(url))
        with open(fname, 'wb') as f: f.write(data)
        duration = time.time() - start
        times.append(duration)
        total_sizes += len(data)
        print(f"[Seq] Download {idx}/{len(urls)} in {duration:.2f}s")
    wall_time = time.time() - wall_start
    sum_time = sum(times)
    total_mb = total_sizes / (1024*1024)
    throughput_wall = total_mb / wall_time if wall_time > 0 else 0
    print(f"[Seq] All done: wall {wall_time:.2f}s, sum durations {sum_time:.2f}s, throughput {throughput_wall:.2f} MB/s")
    return {'mode': 'sequential', 'wall_s': wall_time, 'sum_s': sum_time, 'mb_per_s': throughput_wall}

def download_image_thread(session, url, data_dir, idx, total):
    start = time.time()
    resp = session.get(url, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.content
    fname = os.path.join(data_dir, os.path.basename(url))
    with open(fname, 'wb') as f: f.write(data)
    duration = time.time() - start
    print(f"[Thread] {idx}/{total} in {duration:.2f}s")
    return duration, len(data)

def run_threaded(urls, num_threads, data_dir, timeout):
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir): os.remove(os.path.join(data_dir, f))
    else:
        ensure_data_dir(data_dir)
    session = requests.Session()
    session.verify = False
    total = len(urls)
    wall_start = time.time()
    times, total_sizes = [], 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_image_thread, session, url, data_dir, i+1, total): url for i,url in enumerate(urls)}
        for future in as_completed(futures):
            t, s = future.result()
            times.append(t); total_sizes += s
    wall_time = time.time() - wall_start
    sum_time = sum(times)
    total_mb = total_sizes / (1024*1024)
    throughput_wall = total_mb / wall_time if wall_time > 0 else 0
    print(f"[Thread] All threads done: wall {wall_time:.2f}s, sum durations {sum_time:.2f}s, throughput {throughput_wall:.2f} MB/s")
    return {'mode': f'threaded_{num_threads}thr', 'wall_s': wall_time, 'sum_s': sum_time, 'mb_per_s': throughput_wall}


def plot_comparison(results):
    modes = [r['mode'] for r in results]
    wall = [r['wall_s'] for r in results]
    summ = [r['sum_s'] for r in results]
    mbps = [r['mb_per_s'] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
    ax = axes[0]
    ax.bar(modes, wall, color='lightgreen')
    ax.set_ylabel('Wall-Clock Time (s)'); ax.set_title('Wall-Clock')
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax = axes[1]
    ax.bar(modes, summ, color='skyblue')
    ax.set_ylabel('Sum Durations (s)'); ax.set_title('Sum Durations')
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax = axes[2]
    ax.bar(modes, mbps, color='lightcoral')
    ax.set_ylabel('Throughput (MB/s)'); ax.set_title('Throughput')
    ax.set_xticklabels(modes, rotation=45, ha='right')
    fig.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Vergleich Hybrid, Sequential & Threaded')
    parser.add_argument('--processes', type=int, nargs='+', default=[1,2,4], help='Prozesse')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONC, help='Parallel connections')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS, help='Threads')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Timeout')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Output dir')
    parser.add_argument('--start', type=int, default=1, help='Startseite')
    parser.add_argument('--end', type=int, default=30, help='Endseite')
    args = parser.parse_args()

    urls = generate_image_urls(args.start, args.end)
    ensure_data_dir(args.data_dir)
    results = []
    for p in args.processes:
        results.append(run_hybrid(urls, p, args.concurrency, args.data_dir))
    results.append(run_sequential(urls, args.data_dir, args.timeout))
    results.append(run_threaded(urls, args.threads, args.data_dir, args.timeout))
    plot_comparison(results)

if __name__=='__main__':
    main()
