"""Number plate codes -> [town] by country, for propose_segments.plate_locality: Germany's district codes
(plate_codes_de.py), Romania's county codes and Bulgaria's province codes, each the city or the county's seat."""
from plate_codes_de import PLATE_CODES_DE

PLATE_CODES_RO = {
    'AB': 'Alba Iulia', 'AR': 'Arad', 'AG': 'Pitești', 'BC': 'Bacău', 'BH': 'Oradea', 'BN': 'Bistrița',
    'BT': 'Botoșani', 'BV': 'Brașov', 'BR': 'Brăila', 'B': 'Bucharest', 'BZ': 'Buzău', 'CS': 'Reșița',
    'CL': 'Călărași', 'CJ': 'Cluj-Napoca', 'CT': 'Constanța', 'CV': 'Sfântu Gheorghe', 'DB': 'Târgoviște',
    'DJ': 'Craiova', 'GL': 'Galați', 'GR': 'Giurgiu', 'GJ': 'Târgu Jiu', 'HR': 'Miercurea Ciuc', 'HD': 'Deva',
    'IL': 'Slobozia', 'IS': 'Iași', 'MM': 'Baia Mare', 'MH': 'Drobeta-Turnu Severin', 'MS': 'Târgu Mureș',
    'NT': 'Piatra Neamț', 'OT': 'Slatina', 'PH': 'Ploiești', 'SM': 'Satu Mare', 'SJ': 'Zalău', 'SB': 'Sibiu',
    'SV': 'Suceava', 'TR': 'Alexandria', 'TM': 'Timișoara', 'TL': 'Tulcea', 'VS': 'Vaslui', 'VL': 'Râmnicu Vâlcea',
    'VN': 'Focșani'}
PLATE_CODES_BG = {
    'A': 'Burgas', 'B': 'Varna', 'BH': 'Vidin', 'BP': 'Vratsa', 'BT': 'Veliko Tarnovo', 'E': 'Blagoevgrad',
    'EB': 'Gabrovo', 'EH': 'Pleven', 'K': 'Kardzhali', 'KH': 'Kyustendil', 'M': 'Montana', 'H': 'Shumen',
    'OB': 'Lovech', 'P': 'Ruse', 'PA': 'Pazardzhik', 'PB': 'Plovdiv', 'PK': 'Pernik', 'PP': 'Razgrad', 'C': 'Sofia',
    'CA': 'Sofia', 'CB': 'Sofia', 'CC': 'Silistra', 'CH': 'Sliven', 'CM': 'Smolyan', 'CT': 'Stara Zagora',
    'T': 'Targovishte', 'TX': 'Dobrich', 'X': 'Haskovo', 'Y': 'Yambol'}
PLATE_CODES = {'Germany': {k: v[0] for k, v in PLATE_CODES_DE.items()}, 'Romania': PLATE_CODES_RO,
               'Bulgaria': PLATE_CODES_BG}
# a plate as OCR reads it: the code, then the rest (Germany "HRO AB 123", Romania "B 123 ABC", Bulgaria "CB 6077 MK")
PLATE_FORMS = {'Germany': r'^([A-ZÄÖÜ]{1,3})[\s·.-]{1,2}[A-Z]{1,2}[\s·.-]?\d{1,4}[EH]?$',
               'Romania': r'^([A-Z]{1,2})[\s-]?\d{2,3}[\s-]?[A-Z]{3}$',
               'Bulgaria': r'^([ABCEHKMOPTXY]{1,2})\s?\d{4}\s?[ABCEHKMOPTXY]{2}$'}
