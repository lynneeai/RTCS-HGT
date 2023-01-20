import logging
import time
from datetime import datetime
from urllib.parse import urlparse

from tqdm import tqdm


def program_sleep(sec, print_progress=True):
    if print_progress:
        trange = tqdm(
            range(sec),
            bar_format="sleeping for {n_fmt}/{total_fmt} seconds...",
            leave=False,
        )
        for _ in trange:
            time.sleep(1)
        trange.close()
        print(f"Done sleeping! Slept {sec} seconds!")
    else:
        time.sleep(sec)


def init_logger(log_folder, log_filename="train", timestamp=True):
    """define logging if not already defined"""
    if not logging.getLogger().handlers:
        log_format = "%(asctime)s  %(name)8s  %(levelname)5s  %(message)s"
        log_formatter = logging.Formatter(log_format)
        logging.getLogger().setLevel(logging.INFO)
        # file handler
        if timestamp:
            now_dt = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            log_filename = f"{log_folder}/{log_filename}_{now_dt}.log"
        else:
            log_filename = f"{log_folder}/{log_filename}.log"
        fh = logging.FileHandler(filename=log_filename, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(log_formatter)
        logging.getLogger().addHandler(fh)
        # stream handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(log_formatter)
        logging.getLogger().addHandler(console)
    """---------------------------------"""


def get_url_domain(url, domain_name_only=False):
    parsed_url = urlparse(url)
    url_domain = parsed_url.netloc
    full_url_domain = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    if domain_name_only:
        return clean_url_domain(full_url_domain)
    else:
        url_domain = url_domain.replace("www.", "")
        return url_domain.lower()


def clean_url_domain(url_domain):
    if "://" in url_domain:
        url_domain = url_domain.split("://")[1]
    url_domain = url_domain.replace("www.", "")
    url_components = url_domain.split(".")
    if len(url_components[0]) <= 2:
        url_components.pop(0)
    return url_components[0].lower()
