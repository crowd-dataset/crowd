"""Functions and paths used throughout this project: configs and secrets, country codes, git and email.

Attributes:
    root_dir (str): the top folder of the repo.
    output_dir (str): where generated files go (_output).
    data_dir (str): the data files the analysis reads and writes (analysis_data).
    logger (CustomLogger): this module's logger.
"""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from typing import Dict
import copy
import os
import json
import sys
import pycountry
from custom_logger import CustomLogger
import subprocess
import smtplib
from email.message import EmailMessage

root_dir = os.path.dirname(__file__)
output_dir = os.path.join(root_dir, '_output')
# the data files the analysis reads and writes (not the mapping and metadata files, which stay in the top folder)
data_dir = os.path.join(root_dir, 'analysis_data')

logger = CustomLogger(__name__)  # use custom logger

_MISSING = object()
_parsed = {}  # path -> ((mtime, size), parsed JSON): get_configs is called often (15 times per helper)


def data_path(name: str) -> str:
    """Path of a data file of the analysis: in data_dir, or in the top folder where it still is there (copies of the
    code from before data_dir, e.g. the workers, which get only the .py files on a sync)."""
    path = os.path.join(data_dir, name)
    old = os.path.join(root_dir, name)
    return old if not os.path.exists(path) and os.path.exists(old) else path


def _load_json(file_name: str):
    """The parsed JSON file file_name in root_dir, read again only when its mtime or size changed."""
    path = os.path.join(root_dir, file_name)
    st = os.stat(path)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _parsed.get(path)
    if hit is None or hit[0] != stamp:
        with open(path) as f:
            hit = _parsed[path] = (stamp, json.load(f))
    return hit[1]


def get_secrets(entry_name: str, secret_file_name: str = 'secret') -> Dict[str, str]:
    """Return the entry entry_name of the secrets file (KeyError when it has none)."""
    with open(os.path.join(root_dir, secret_file_name)) as f:
        return json.load(f)[entry_name]


def get_configs(entry_name: str, config_file_name: str = 'config', config_default_file_name: str = 'default.config',
                default=_MISSING):
    """
    Return the entry entry_name of the config file; default when it has none (KeyError when no default is given).
    Exits when the config file is missing, badly formatted or has fewer entries than default.config.
    """
    if not check_config(config_file_name, config_default_file_name):
        sys.exit()
    value = _load_json(config_file_name).get(entry_name, default)
    if value is _MISSING:
        raise KeyError(entry_name)
    return copy.deepcopy(value)  # a fresh copy each call, as when the file was read every time


def check_config(config_file_name: str = 'config',
                 config_default_file_name: str = 'default.config'):
    """True when the config file and default.config can be read and the config has at least as many entries."""
    loaded = []
    for name in (config_file_name, config_default_file_name):
        try:
            loaded.append(_load_json(name))
        except FileNotFoundError:
            logger.error('Config file {} not found.', name)
            return False
        except json.decoder.JSONDecodeError:
            logger.error('Config file {} badly formatted. Please update based on {}.', name, config_default_file_name)
            return False
    config, default = loaded
    if len(config) < len(default):
        logger.error('Config file has {} variables, which is fewer than {} variables in {}. Please update.',
                     len(config), len(default), config_default_file_name)
        return False
    return True


def correct_country(country):
    """
    Corrects common country name variations for compatibility with pycountry.countries.get(name=...).

    Args:
        country (str): Name of country in its full form.

    Returns:
        str: Corrected country.
    """
    corrections = {
        'Russia': 'Russian Federation',
        'Syria': 'Syrian Arab Republic',
        'South Korea': 'Korea, Republic of',
        'North Korea': "Korea, Democratic People's Republic of",
        'Korea': 'Korea, Republic of',
        'Iran': 'Iran, Islamic Republic of',
        'Vietnam': 'Viet Nam',
        'Venezuela': 'Venezuela, Bolivarian Republic of',
        'Bolivia': 'Bolivia, Plurinational State of',
        'Moldova': 'Moldova, Republic of',
        'Laos': "Lao People's Democratic Republic",
        'Brunei': 'Brunei Darussalam',
        'Czech Republic': 'Czechia',
        'Ivory Coast': "Côte d'Ivoire",
        'Cape Verde': 'Cabo Verde',
        'Swaziland': 'Eswatini',
        'Macau': 'Macao',
        'Taiwan': 'Taiwan, Province of China',
        'Tanzania': 'Tanzania, United Republic of',
        'UK': 'United Kingdom',
        'Palestine': 'Palestine, State of',
        'Micronesia': 'Micronesia, Federated States of',
        'São Tomé and Príncipe': 'Sao Tome and Principe',
        'Turkiye': 'Türkiye',  # pycountry's name (24.6.1 has no "Turkey", "Bahamas, The" or "Holy See")
        'Turkey': 'Türkiye',
        'Congo (Democratic Republic)': 'Congo, The Democratic Republic of the',
        'Congo (Congo-Brazzaville)': 'Congo',
        'Burma': 'Myanmar',
        'East Timor': 'Timor-Leste',
        'Saint Kitts': 'Saint Kitts and Nevis',
        'Saint Vincent': 'Saint Vincent and the Grenadines',
        'Antigua': 'Antigua and Barbuda',
        'Trinidad': 'Trinidad and Tobago',
        'Slovak Republic': 'Slovakia',
        'Vatican': 'Holy See (Vatican City State)',
        'British Virgin Islands': 'Virgin Islands, British',
        'United States Virgin Islands': 'Virgin Islands, U.S.',
        'Falkland Islands': 'Falkland Islands (Malvinas)',
        'Saint Helena': 'Saint Helena, Ascension and Tristan da Cunha',
        'Saint Martin': 'Saint Martin (French part)',
        'Sint Maarten': 'Sint Maarten (Dutch part)',
    }

    return corrections.get(country, country)


def _country(country_name):
    """pycountry's country of that name, None when it knows none."""
    try:
        return pycountry.countries.get(name=country_name)
    except (KeyError, LookupError):
        return None


def get_iso2_country_code(country_name):
    """ISO-2 code of a country given by its full name (pycountry's, see correct_country); None when unknown."""
    if country_name == 'Kosovo':
        return 'XK'
    country = _country(country_name)
    return country.alpha_2 if country else None


def get_iso3_country_code(country_name):
    """ISO-3 code of a country given by its full name (pycountry's, see correct_country); None when unknown."""
    if country_name == 'Kosovo':
        return 'XKX'
    country = _country(country_name)
    return country.alpha_3 if country else None


def git_pull():
    """git pull changes from the repo with a terminal command."""
    try:
        logger.info("Attempting to pull latest changes from git repository...")
        result = subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
        logger.info(f"Git pull successful:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git pull failed with error:\n{e.stderr}")


def send_email(subject, content, sender, recipients):
    """
    Send email with certain subject and content from sender to recipients (a list). A failure is logged with its
    traceback, not raised: a pass of the pipeline must not stop for a mail.
    """
    msg = EmailMessage()
    msg.set_content(content)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    try:
        with smtplib.SMTP_SSL(get_secrets("email_smtp"), 465) as smtp:
            smtp.login(get_secrets("email_account"), get_secrets("email_password"))
            smtp.send_message(msg)
            logger.info(f"Sent email to: {recipients}")
    except Exception as e:
        logger.error(f"Failed to send email '{subject}' to {recipients}: {e!r}", exc_info=True)
