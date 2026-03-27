import pandas as pd
import numpy as np
import datetime
from dateutil import easter

def get_holidays(years):
    """
    Generates a set of Portuguese national holiday dates for a list of given years.
    """
    holidays = set()
    for year in years:
        e_day = easter.easter(year)
        good_friday = e_day - datetime.timedelta(days=2)
        corpus_christi = e_day + datetime.timedelta(days=60)
        carnaval = e_day - datetime.timedelta(days=47)

        
        holidays.update([
            datetime.date(year, 1, 1),    # Ano Novo (New Year's Day)
            datetime.date(year, 4, 25),   # Dia da Liberdade (Freedom Day)
            datetime.date(year, 5, 1),    # Dia do Trabalhador (Labor Day)
            datetime.date(year, 6, 10),   # Dia de Portugal (Portugal Day)
            datetime.date(year, 8, 15),   # Assunção de Nossa Senhora (Assumption of Mary)
            datetime.date(year, 12, 8),   # Imaculada Conceição (Immaculate Conception)
            datetime.date(year, 12, 25),  # Natal (Christmas Day)
            good_friday,                  # Sexta-feira Santa (Good Friday)
            e_day,                        # Páscoa (Easter Sunday)
            carnaval                      # Carnaval (Shrove Tuesday: Unofficial but heavily observed)
        ])

        # Holidays suspended by the government (Troika) between 2013 and 2015
        # More infos at https://www.theguardian.com/world/2016/jan/08/portugals-socialist-government-restores-holidays-cut-during-austerity-drive
        if year not in [2013, 2014, 2015]:
            holidays.update([
                datetime.date(year, 10, 5),   # Implantação da República (Republic Day)
                datetime.date(year, 11, 1),   # Dia de Todos os Santos (All Saints' Day)
                datetime.date(year, 12, 1),   # Restauração da Independência (Restoration of Independence)
                corpus_christi                # Corpo de Deus (Corpus Christi)
            ])


    return holidays