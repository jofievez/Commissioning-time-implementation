import pandas as pd
import os
from pathlib import Path
import sobol
import numpy as np

class AmplUQ:

    """

    The AmplUQ class allows to generate samples of uncertain parameters and
    change the values of the affected parameters of the ampl_object, run the
    pathway optimisation and return the total transition cost

    Parameters
    ----------
    ampl_obj : AmplObject object of ampl_object module
        Ampl object containing the optimisation problem and its attributes
    uq_file : string
        Path towards the file containing the uncertainty range

    """

    def __init__(self, ampl_obj):

        self.ampl_obj = ampl_obj
        project_path = Path(__file__).parents[1]
        self.uq_file = os.path.join(project_path,'uncertainties','uc_final.xlsx')

    def transcript_unexpected_events(self, uncer_params, years_wnd, technologies, resources):

        up = uncer_params

        c_op = self.ampl_obj.get_elem('c_op', type_of_elem='Param').copy()
        c_inv = self.ampl_obj.get_elem('c_inv', type_of_elem='Param').copy()
        avail = self.ampl_obj.get_elem('avail', type_of_elem='Param').copy()
        f_max = self.ampl_obj.get_elem('f_max', type_of_elem='Param').copy()
        f_min = self.ampl_obj.get_elem('f_min', type_of_elem='Param').copy()
        end_uses_demand_year = self.ampl_obj.get_elem('end_uses_demand_year', type_of_elem='Param').copy()

        # round nuclear impacts
        up['nuclear'] = min((up['nuclear'] * 5 // 1) / 4, 1.)

        # determine how far installed capacity is from upper limit
        current_pv = f_max.at[('YEAR_2030', 'PV'), 'f_max']
        fmax_pv = 104.1
        current_w_on = f_max.at[('YEAR_2030', 'WIND_ONSHORE'), 'f_max']
        fmax_w_on = 20.
        current_w_off = f_max.at[('YEAR_2030', 'WIND_OFFSHORE'), 'f_max']
        fmax_w_off = 8.

        slope_pv = fmax_pv - current_pv
        slope_w_on = fmax_w_on - current_w_on
        slope_w_off = fmax_w_off - current_w_off

        dem_cum = 0.
        cost_cum = 0.

        # round impacts
        for idx, yri in enumerate([2035,2040,2045,2050]):
            yr = 'YEAR_%i' %yri
            up['geo_tension_efuels_%i' %yri] = min((up['geo_tension_efuels_%i' %yri] * 6 // 1) / 5, 1.)
            up['geo_tension_biofuels_%i' %yri] = min((up['geo_tension_biofuels_%i' %yri] * 6 // 1) / 5, 1.)
            up['geo_tension_elec_%i' %yri] = min((up['geo_tension_elec_%i' %yri] * 6 // 1) / 5, 1.)
            up['avail_pv_%i' % yri] = min((up['avail_pv_%i' % yri] * 6 // 1) / 5, 1.)
            up['avail_wind_on_%i' % yri] = min((up['avail_wind_on_%i' % yri] * 6 // 1) / 5, 1.)
            up['avail_wind_off_%i' % yri] = min((up['avail_wind_off_%i' % yri] * 6 // 1) / 5, 1.)
            up['exchange_rate_%i' % yri] = min((up['exchange_rate_%i' % yri] * 6 // 1) / 5, 1.)
            up['demand_%i' % yri] = min((up['demand_%i' % yri] * 6 // 1) / 5, 1.)
            up['limit_renovation_%i' % yri] = min((up['limit_renovation_%i' % yri] * 6 // 1) / 5, 1.)
            up['limit_mob_change_%i' % yri] = min((up['limit_mob_change_%i' % yri] * 6 // 1) / 5, 1.)

            # determine increase in available installed capacity to install for PV and wind
            actual_slope_pv = max(slope_pv * (1. - up['avail_pv_%i' %yri])**2., 0.)
            actual_slope_w_on = max(slope_w_on * (1. - up['avail_wind_on_%i' %yri])**2., 0.)
            actual_slope_w_off = max(slope_w_off * (1. - up['avail_wind_off_%i' %yri])**2., 0.)

            idx += 1
            current_pv += actual_slope_pv
            if current_pv > fmax_pv:
                current_pv = fmax_pv
            self.ampl_obj.set_params('f_max', f_max.loc[(yr, ['PV']), :] / fmax_pv * current_pv)

            current_w_on += actual_slope_w_on
            if current_w_on > fmax_w_on:
                current_w_on = fmax_w_on
            self.ampl_obj.set_params('f_max', f_max.loc[(yr, ['WIND_ONSHORE']), :] / fmax_w_on * current_w_on)

            current_w_off += actual_slope_w_off
            if current_w_off > fmax_w_off:
                current_w_off = fmax_w_off
            self.ampl_obj.set_params('f_max', f_max.loc[(yr, ['WIND_OFFSHORE']), :] / fmax_w_off * current_w_off)

            # increase investment cost of imported technologies (from Asia) due to deterioration exchange rate
            cost_cum += (up['exchange_rate_%i' %yri] * 0.4)
            cost_multiplier = 1. + cost_cum
            technologies_from_asia = [
                'PV',
                'BATT_LI',
                'BEV_BATT',
                'PHEV_BATT',
                'CAR_BEV',
                'CAR_PHEV',
                'CAR_HEV',
                'TRUCK_ELEC',
                'DHN_SOLAR',
                'DEC_SOLAR',
                'IND_DIRECT_ELEC',
                'DEC_DIRECT_ELEC',
                'DHN_HP_ELEC',
                'DEC_HP_ELEC',
                'H2_ELECTROLYSIS',
                ]
            for tech in technologies_from_asia:
                self.ampl_obj.set_params('c_inv', c_inv.loc[(yr, [tech]), :] * cost_multiplier)

            # increase energy demand in all sectors
            dem_cum += (up['demand_%i' %yri] * 0.15)
            demand_multiplier = 1. + dem_cum
            typess = [
            'ELECTRICITY',
            'HEAT_HIGH_T',
            'HEAT_LOW_T_HW',
            'HEAT_LOW_T_SH',
            'LIGHTING',
            'MOBILITY_FREIGHT',
            'MOBILITY_PASSENGER',
            'NON_ENERGY',
            ]
            for typesss in typess:
                self.ampl_obj.set_params('end_uses_demand_year',
                                         end_uses_demand_year.loc[(yr,[typesss],['HOUSEHOLDS']),:] * demand_multiplier)
                self.ampl_obj.set_params('end_uses_demand_year',
                                         end_uses_demand_year.loc[(yr,[typesss],['SERVICES']),:] * demand_multiplier)
                self.ampl_obj.set_params('end_uses_demand_year',
                                         end_uses_demand_year.loc[(yr,[typesss],['INDUSTRY']),:] * demand_multiplier)
                self.ampl_obj.set_params('end_uses_demand_year',
                                         end_uses_demand_year.loc[(yr,[typesss],['TRANSPORTATION']),:] * demand_multiplier)

        # impact the availability of unicorn technologies
        avails = [
            'avail_nuclear_smr',
            'avail_ccs',
            'avail_dac',
            'avail_geoth',
            'avail_ccgt_nh3',
            'avail_nh3_cracking',
            ]

        for av in avails:
            up[av] = min((up[av] * 4 // 1) / 3, 1.)

        nuclear_value = up['nuclear']
        nuclear_smr_value = up['avail_nuclear_smr']
        ccs_value = up['avail_ccs']
        dac_value = up['avail_dac']
        geoth_value = up['avail_geoth']
        ccgt_nh3_value = up['avail_ccgt_nh3']
        ccgt_nh3_cracking_value = up['avail_nh3_cracking']

        ups = {}
        # Initialize values for other technologies
        ups.update({f'nuclear_{year}': 0.00001 for year in range(2035, 2055, 5)})
        ups.update({f'nuclear_smr_{year}': 0.00001 for year in range(2035, 2055, 5)})
        ups.update({f'dac_{year}': 0.00001 for year in range(2035, 2055, 5)})
        ups.update({f'ccs_{year}': 0.00001 for year in range(2035, 2055, 5)})
        ups.update({f'geoth_{year}': 0.00001 for year in range(2035, 2055, 5)})
        ups.update({f'ccgt_nh3_{year}': 0.00001 for year in range(2035, 2055, 5)})
        ups.update({f'nh3_cracking_{year}': 0.00001 for year in range(2035, 2055, 5)})

        # Define common thresholds
        nuclear_years = [2050, 2045, 2040, 2035]
        nuclear_thresholds = [0.25, 0.5, 0.75, 1]


        # Define years for nuclear and reversed years for other technologies
        common_years = [2040, 2045, 2050]
        common_thresholds = [0.33, 0.66, 1]

        # Define thresholds and corresponding years for each technology
        tech_thresholds = {
            'nuclear': (nuclear_thresholds, nuclear_years),
            'dac': (common_thresholds, common_years),
            'nuclear_smr': (common_thresholds, common_years),
            'ccs': (common_thresholds, common_years),
            'geoth': (common_thresholds, common_years),
            'ccgt_nh3': (common_thresholds, common_years),
            'nh3_cracking': (common_thresholds, common_years)
        }

        # Define a dictionary for technology values
        tech_values = {
            'nuclear': nuclear_value,
            'nuclear_smr': nuclear_smr_value,
            'dac': dac_value,
            'ccs': ccs_value,
            'geoth': geoth_value,
            'ccgt_nh3': ccgt_nh3_value,
            'nh3_cracking': ccgt_nh3_cracking_value
        }

        # Update ups based on each technology’s threshold and value—so when they become available
        for tech, (thresholds, years) in tech_thresholds.items():
            value = tech_values[tech]
            for threshold, year in zip(thresholds, years):  # Skip the first threshold as it leaves values at 0
                if value < threshold:
                    ups[f'{tech}_{year}'] = 0.999999999

        # # Override for DAC and CCS if value > 0.5
        for tech in ['dac', 'ccs']:
            value = tech_values[tech]
            if value < 0.5:
                for year in tech_thresholds[tech][1]:  # Get years for that tech
                    ups[f'{tech}_{year}'] = 0.99999999
            else:
                for year in tech_thresholds[tech][1]:  # Get years for that tech
                    ups[f'{tech}_{year}'] = 0.00001

        for threshold, year in zip(*tech_thresholds['ccs']):
            dac_key = f'dac_{year}'
            ccs_key = f'ccs_{year}'
            if dac_key in ups:
                ups[ccs_key] = ups[dac_key]

        # overwrite availability of resources and resistances to change for the target years
        for y in years_wnd:

            self.ampl_obj.set_params('avail',avail.loc[(y,['ELECTRICITY']),:] * (1. - up.loc['geo_tension_elec_%s' %y[-4:]]))
            self.ampl_obj.set_params('c_op', c_op.loc[(y,['ELECTRICITY']),:] * (1. + up.loc['geo_tension_elec_%s' %y[-4:]]))

            self.ampl_obj.set_params('avail',avail.loc[(y,['BIODIESEL']),:]* (1. - up.loc['geo_tension_biofuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('avail',avail.loc[(y,['BIOETHANOL']),:]* (1. - up.loc['geo_tension_biofuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('avail',avail.loc[(y,['H2_RE']),:]* (1. - up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('avail',avail.loc[(y,['GAS_RE']),:]* (1. - up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('avail',avail.loc[(y,['METHANOL_RE']),:]* (1. - up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('avail',avail.loc[(y,['AMMONIA_RE']),:]* (1. - up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('c_op', c_op.loc[(y, ['H2_RE']), :] * (1. + up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('c_op', c_op.loc[(y, ['GAS_RE']), :] * (1. + up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('c_op', c_op.loc[(y, ['METHANOL_RE']), :] * (1. + up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('c_op', c_op.loc[(y, ['AMMONIA_RE']), :] * (1. + up.loc['geo_tension_efuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('c_op', c_op.loc[(y, ['BIODIESEL']), :] * (1. + up.loc['geo_tension_biofuels_%s' %y[-4:]]))
            self.ampl_obj.set_params('c_op', c_op.loc[(y, ['BIOETHANOL']), :] * (1. + up.loc['geo_tension_biofuels_%s' %y[-4:]]))

            self.ampl_obj.set_params('limit_LT_renovation', {(y): 0.33 * (1. - up['limit_renovation_%s' %y[-4:]]) })
            self.ampl_obj.set_params('limit_pass_mob_changes', {(y): 0.5 * (1. - up['limit_mob_change_%s' %y[-4:]]) })
            self.ampl_obj.set_params('limit_freight_changes', {(y):  0.5 * (1. - up['limit_mob_change_%s' %y[-4:]]) })

            self.ampl_obj.set_params('f_max', f_max.loc[(y, ['NUCLEAR']), :] * ups['nuclear_%s' %y[-4:]])
            self.ampl_obj.set_params('f_min', f_min.loc[(y, ['NUCLEAR']), :] * ups['nuclear_%s' %y[-4:]])
            self.ampl_obj.set_params('f_max', f_max.loc[(y, ['NUCLEAR_SMR']), :] * ups['nuclear_smr_%s' %y[-4:]])

            self.ampl_obj.set_params('f_max', f_max.loc[(y, ['GEOTHERMAL']), :] * ups['geoth_%s' %y[-4:]])

            if ups['ccs_%s' %y[-4:]] < 0.5:
                ab = f_max.loc[(y, ['INDUSTRY_CCS']), :]
                ab.at[(y, 'INDUSTRY_CCS'), 'f_max'] = 0.000001
                self.ampl_obj.set_params('f_max', ab)

            if ups['dac_%s' %y[-4:]] < 0.5:
                ab = f_max.loc[(y, ['ATM_CCS']), :]
                ab.at[(y, 'ATM_CCS'), 'f_max'] = 0.000001
                self.ampl_obj.set_params('f_max', ab)

            if ups['ccgt_nh3_%s' %y[-4:]] < 0.5:
                ab = f_max.loc[(y, ['CCGT_AMMONIA']), :]
                ab.at[(y, 'CCGT_AMMONIA'), 'f_max'] = 0.000001
                self.ampl_obj.set_params('f_max', ab)

            if ups['nh3_cracking_%s' %y[-4:]] < 0.5:
                ab = f_max.loc[(y, ['AMMONIA_TO_H2']), :]
                ab.at[(y, 'AMMONIA_TO_H2'), 'f_max'] = 0.000001
                self.ampl_obj.set_params('f_max', ab)