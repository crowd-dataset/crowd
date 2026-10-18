import common
import pandas as pd
import pycountry
from custom_logger import CustomLogger


logger = CustomLogger(__name__)  # use custom logger


class ISO:
    def __init__(self) -> None:
        pass

    def iso2_to_flag(self, iso2):
        """
        Convert a 2-letter ISO country code to a corresponding emoji flag.

        This uses Unicode regional indicator symbols to represent flags. For example, 'US' becomes 🇺🇸.

        Args:
            iso2 (str or None): Two-letter ISO country code (e.g., 'US'). If None, returns a placeholder.

        Returns:
            str: A flag emoji corresponding to the country code, or 🇽🇰 (Kosovo) as a fallback.
        """
        logger.debug(f"Converting iso2 {iso2} to flag.")

        # Special case for Kosovo
        # todo: handle Kosovo ISO2
        if not iso2:
            return "🇽🇰"

        if not iso2 or len(iso2) != 2 or not iso2.isalpha():
            return "🏳️"
        iso2 = iso2.upper()

        # Load the CSV - ideally, you should cache this in self.flag_data!
        flag_data = pd.read_csv(common.data_path('countries.csv'), dtype=str)

        # Ensure column names are stripped of whitespace
        flag_data.columns = flag_data.columns.str.strip()

        # Find the row where iso2 matches
        result = flag_data.loc[flag_data['iso2'].str.upper() == iso2, 'flag']

        if not result.empty and isinstance(result.iloc[0], str) and result.iloc[0].strip():  # type: ignore
            return result.iloc[0]  # type: ignore

        # Fallback: Unicode flag emoji from iso2 code
        try:
            return chr(0x1F1E6 + ord(iso2[0]) - ord('A')) + chr(0x1F1E6 + ord(iso2[1]) - ord('A'))
        except Exception:
            return "🏳️"

    def iso3_to_iso2(self, iso3_code):
        """
        Convert an ISO 3166-1 alpha-3 country code (e.g., 'USA') to an ISO 3166-1 alpha-2 code (e.g., 'US').

        Args:
            iso3_code (str): The three-letter country code (ISO alpha-3).

        Returns:
            str or None: The corresponding two-letter country code (ISO alpha-2),
                         or None if the code is invalid or not found.
        """
        try:
            # Look up the country using pycountry
            country = pycountry.countries.get(alpha_3=iso3_code)
            return country.alpha_2 if country else None
        except (AttributeError, LookupError) as e:
            logger.debug(f"Converting up ISO-3 {iso3_code} to ISO-2 returned error: {e}.")
            return None
