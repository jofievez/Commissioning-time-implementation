import rheia.POST_PROCESS.post_process as rheia_pp
import rheia.UQ.uncertainty_quantification as rheia_uq
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import plotly.io as pio
import numpy as np
import os, sys
import re
import copy
from pathlib import Path
import pickle as pkl
import hashlib

pylibPath = os.path.abspath("../pylib")
if pylibPath not in sys.path:
    sys.path.insert(0, pylibPath)

from ampl_graph import AmplGraph

pio.templates.default = 'simple_white'
pio.kaleido.scope.mathjax = None

pio.renderers.default = 'browser'

class AmplUQGraph:

    """

    The AmplUQGraph class allows to plot the relevant outputs (e.g. installed capacitites, used resources, costs)
    of the different samples of UQ on an optimisation problem.

    Parameters
    ----------
    result_list: list(Pandas.DataFrame)
        Unpickled list where relevant outputs has been stored

    """

    def __init__(self, case_study, ampl_obj,ref_case=None,smr_case=None,result_dir_comp = [], pol_order=2):
        self.result_dir = result_dir_comp
        self.case = 'UNC_ANAL_ES_PATHWAY'
        self.pol_order = pol_order
        self.my_post_process_uq = rheia_pp.PostProcessUQ(self.case,self.pol_order)
        self.case_study = case_study
        self.case_study_dir_path = str(Path(self.my_post_process_uq.result_path).absolute() / self.case_study)
        
        self.objective = 'cost'
        self.threshold = 1
        self.threshold_filter = 0.01

        # Updated x_axis with all available years (19 years instead of 7)
        self.x_axis = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
        #self.x_axis = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
        # In 2020, 37.41% of electricity in EU-27 was produced from renewables (https://ec.europa.eu/eurostat/databrowser/view/NRG_IND_REN__custom_4442440/default/table?lang=en)
        self.re_share_elec = np.linspace(0.3741,1,len(self.x_axis))
        
        self.gather_results()
        self.ampl_uq_collector = self.unpkl(self)
        self.samples_file = os.path.join(self.case_study_dir_path,'samples.csv')
        samples = pd.read_csv(self.samples_file)
        samples.index.name = 'Sample'
        self.ampl_uq_collector['Samples'] = samples
        self.ampl_obj = ampl_obj
        project_path = Path(__file__).parents[1]
        uq_file = os.path.join(project_path,'uncertainties','uc_final.xlsx')
        uncert_param = pd.read_excel(uq_file,sheet_name='Parameters',engine='openpyxl',index_col=0)
        uncert_param_meaning = uncert_param['Meaning_short']
        self.uncert_param_meaning = dict(uncert_param_meaning)
        
        #uncert_range = pd.read_excel(uq_file,sheet_name='YEAR_2025',engine='openpyxl',index_col=0)
        #self.uncert_nominal = self.get_nominal(uncert_range)
        
        # Charger ref_case si fourni
        self.ref_case = ref_case
        self.ref_results = None
        if ref_case is not None:
            self.ref_file = os.path.join(project_path,'out',ref_case,'_Results.pkl')
            ref_results = open(self.ref_file,"rb")
            self.ref_results = pkl.load(ref_results)
            ref_results.close()
        
        # Charger smr_case si fourni
        self.smr_case = smr_case
        self.smr_results = None
        if smr_case is not None:
            self.smr_file = os.path.join(project_path,'out',smr_case,'_Results.pkl')
            smr_results = open(self.smr_file,"rb")
            self.smr_results = pkl.load(smr_results)
            smr_results.close()

        # Keep unfiltered copies so year focus can be changed any time.
        self._ampl_uq_collector_full = self._copy_container(self.ampl_uq_collector)
        self._ref_results_full = self._copy_container(self.ref_results)
        self._smr_results_full = self._copy_container(self.smr_results)
        self.apply_year_focus()
        
        self.color_dict_full = self._dict_color_full() if ampl_obj is not None else {}
        
        self.outdir = os.path.join(self.case_study_dir_path,'graphs/')
        if not os.path.exists(Path(self.outdir)):
            Path(self.outdir).mkdir(parents=True,exist_ok=True)

        self.category = self._group_sets() if ampl_obj is not None else None

    @staticmethod
    def _extract_year(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        match = re.search(r'(?:19|20)\d{2}', str(value))
        return int(match.group(0)) if match else None

    @staticmethod
    def _copy_container(container):
        if isinstance(container, dict):
            return {k: AmplUQGraph._copy_container(v) for k, v in container.items()}
        if isinstance(container, (pd.DataFrame, pd.Series)):
            return container.copy(deep=True)
        return container

    def _filter_df_or_series_by_years(self, obj, years_set):
        if obj is None:
            return None

        if isinstance(obj, pd.DataFrame):
            if obj.empty:
                return obj.copy()
            if isinstance(obj.index, pd.MultiIndex) and 'Years' in obj.index.names:
                mask = [self._extract_year(v) in years_set for v in obj.index.get_level_values('Years')]
                return obj.loc[mask].copy()
            if obj.index.name == 'Years':
                mask = [self._extract_year(v) in years_set for v in obj.index]
                return obj.loc[mask].copy()
            if 'Years' in obj.columns:
                mask = obj['Years'].map(lambda v: self._extract_year(v) in years_set)
                return obj.loc[mask].copy()
            return obj.copy()

        if isinstance(obj, pd.Series):
            if obj.empty:
                return obj.copy()
            if isinstance(obj.index, pd.MultiIndex) and 'Years' in obj.index.names:
                mask = [self._extract_year(v) in years_set for v in obj.index.get_level_values('Years')]
                return obj.loc[mask].copy()
            if obj.index.name == 'Years':
                mask = [self._extract_year(v) in years_set for v in obj.index]
                return obj.loc[mask].copy()
            return obj.copy()

        return obj

    def _filter_container_by_years(self, container, years_set):
        if isinstance(container, dict):
            return {k: self._filter_container_by_years(v, years_set) for k, v in container.items()}
        return self._filter_df_or_series_by_years(container, years_set)

    def apply_year_focus(self):
        """Apply current self.x_axis years to all loaded results.

        Usage:
        1) set self.x_axis, e.g. [2020, 2021, ..., 2027]
        2) call self.apply_year_focus()
        """
        years_set = set(int(y) for y in self.x_axis)
        self.ampl_uq_collector = self._filter_container_by_years(self._ampl_uq_collector_full, years_set)
        self.ref_results = self._filter_container_by_years(self._ref_results_full, years_set)
        self.smr_results = self._filter_container_by_years(self._smr_results_full, years_set)

    def _export_corr_heatmap(self, row_matrix, x_labels, y_labels, title, zmin, zmax,
                             out_dir, filename_stem, decimals=2, interpolation='bicubic',
                             text_threshold=None, bad_color=None, xlabel_rotation=-45,
                             fig_height=None):
        """Export a correlation heatmap (any shape) via matplotlib (no Kaleido)."""
        gray = '#5a5a5a'
        mat = row_matrix.values.astype(float)
        n_rows, n_cols = mat.shape

        cell = 0.75
        fig_w = max(5, n_cols * cell + 3)
        fig_h = fig_height if fig_height is not None else max(2.2, n_rows * cell + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap = copy.copy(plt.cm.RdBu_r)
        if bad_color is not None:
            cmap.set_bad(bad_color)
        display_mat = mat.copy()
        if text_threshold is not None:
            display_mat[np.abs(display_mat) < text_threshold] = 0.0
        im = ax.imshow(display_mat, aspect='auto', cmap=cmap, vmin=zmin, vmax=zmax,
                       interpolation=interpolation)

        for ri in range(n_rows):
            for ci in range(n_cols):
                v = mat[ri, ci]
                if np.isnan(v):
                    continue
                if text_threshold is not None and abs(v) < text_threshold:
                    continue
                txt_color = 'white' if abs(v) > 0.6 else gray
                ax.text(ci, ri, f'{v:.{decimals}f}', ha='center', va='center',
                        fontsize=8, color=txt_color, fontweight='bold')

        ax.set_xticks(range(n_cols))
        ha = 'center' if xlabel_rotation == 0 else 'left'
        ax.set_xticklabels(x_labels, fontsize=9, color=gray, rotation=xlabel_rotation, ha=ha)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_labels, fontsize=9, color=gray)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02, aspect=max(10, n_rows * 3))
        cbar.set_ticks([zmin, 0, zmax])
        cbar.set_ticklabels([f'{zmin:.0f}', '0', f'{zmax:.0f}'])
        cbar.ax.tick_params(labelsize=9, colors=gray)
        cbar.outline.set_visible(False)

        ax.set_title(title, fontsize=13, color=gray, pad=10, fontweight='bold')
        fig.tight_layout()

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir + filename_stem + '.pdf', bbox_inches='tight')
        fig.savefig(out_dir + filename_stem + '.svg', bbox_inches='tight')
        plt.close(fig)

    def _match_rows_by_sample_strict(self, df, sample_col, sample_id):
        rows = df.loc[df[sample_col].astype(str) == str(sample_id)]
        if not rows.empty:
            return rows, 'string-exact'

        sid_num = pd.to_numeric(pd.Series([sample_id]), errors='coerce').iloc[0]
        col_num = pd.to_numeric(df[sample_col], errors='coerce')
        if pd.notna(sid_num):
            rows = df.loc[col_num == sid_num]
            if not rows.empty:
                return rows, 'numeric-exact'

        return rows, 'not-found'

    def _get_min_max_samples_from_cost(self, ampl_uq_collector=None):
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        samples_params = ampl_uq_collector.get('Samples')
        if not isinstance(samples_params, pd.DataFrame) or samples_params.empty:
            return None, None, samples_params
        if self.objective not in samples_params.columns:
            return None, None, samples_params

        costs = pd.to_numeric(samples_params[self.objective], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(costs).any():
            return None, None, samples_params

        min_row_zero_based = int(np.nanargmin(costs))
        max_row_zero_based = int(np.nanargmax(costs))
        # Business rule: sample N corresponds to CSV row N+1 (header at line 1).
        min_sample = min_row_zero_based + 1
        max_sample = max_row_zero_based + 1
        return min_sample, max_sample, samples_params

    def _load_deterministic_result(self, key):
        det_paths = [
            "/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/UQ/Deterministe/Runs/_Results.pkl",
            "/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/UQ/Deterministe/Runs/Deterministe/_Results.pkl",
        ]
        det_data = None
        det_pkl_path = det_paths[0]
        for candidate_path in det_paths:
            det_pkl_path = candidate_path
            try:
                with open(candidate_path, 'rb') as f:
                    det_results = pkl.load(f)
                if key in det_results:
                    det_data = det_results[key].copy()
                break
            except Exception:
                continue
        return det_data, det_pkl_path
    
    def get_spec_output(self,dict_uq,output,element,year,focus='High',calc_Sobol=False, flip = True):
        
        labels = [None] * len(output)
        samples_plus = self.ampl_uq_collector['Samples'].copy()
        col_objective = samples_plus.columns.get_loc(self.objective)
        samples_plus = samples_plus.iloc[:,:col_objective+1]
        result_ref_full = dict()
        result_smr_full = dict()
        meaning_output = self.dict_meaning()
        for i in range(len(output)):
            
            nom_values = pd.DataFrame(index=samples_plus.columns,columns=['Nominal'],data=0)
            nom_values.index.name='Parameter'
            nom_values.update(self.uncert_nominal)
            
            nom_values_ref = pd.DataFrame(index=samples_plus.columns,columns=['REF'],data=0)
            nom_values_ref.index.name='Parameter'
            
            nom_values_smr = pd.DataFrame(index=samples_plus.columns,columns=['SMR'],data=0)
            nom_values_smr.index.name='Parameter'
            
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
                layer = element[i][1]
            elif out == 'TotalGwp':
                el = element[i]
            y = year[i]
            label = out+'_'+el+'_'+y
            labels[i] = label
            if out == 'F':
                results = self.ampl_uq_collector['Assets'][['F','Sample']]
                results = results.loc[results.index.get_level_values('Technologies') == el]
                results.rename(columns = {out:label},inplace=True)
                
                result_ref = self.ref_results['Assets']['F']
                result_ref = result_ref.loc[result_ref.index.get_level_values('Technologies') == el]
                
                result_smr = self.smr_results['Assets']['F']
                result_smr = result_smr.loc[result_smr.index.get_level_values('Technologies') == el]
                
            elif out == 'Ft':
                results = self.ampl_uq_collector['Year_balance'][[layer,'Sample']]
                results = results.loc[results.index.get_level_values('Elements') == el]
                results.rename(columns = {layer:label},inplace=True)
                
                result_ref = self.ref_results['Year_balance'][layer]
                result_ref = result_ref.loc[result_ref.index.get_level_values('Elements') == el]
                
                result_smr = self.smr_results['Year_balance'][layer]
                result_smr = result_smr.loc[result_smr.index.get_level_values('Elements') == el]
            
            elif out == 'TotalGwp':
                results = self.ampl_uq_collector['TotalGwp'][['TotalGWP','Sample']]
                results.rename(columns = {'TotalGWP':label},inplace=True)
                result_ref = self.ref_results['TotalGwp']['TotalGWP']
                result_smr = self.smr_results['TotalGwp']['TotalGWP']
                results.index.name = 'Years'
                result_ref.index.name = 'Years'
                result_smr.index.name = 'Years'
                
                
            results = results.loc[results.index.get_level_values('Years') == y]
            result_ref = result_ref.loc[result_ref.index.get_level_values('Years') == y]
            result_smr = result_smr.loc[result_smr.index.get_level_values('Years') == y]
            
            if result_ref.empty:
                result_ref_full[label] = 0
            else:
                result_ref_full[label] = result_ref.values[0]
            
            if result_smr.empty:
                result_smr_full[label] = 0
            else:
                result_smr_full[label] = result_smr.values[0]

            results.reset_index(inplace=True)
            results = results.set_index(['Sample'])
            samples_plus[label] = results[label]
            samples_plus.fillna(0,inplace=True)
        
        samples_plus.to_csv(self.samples_file,index=False)
        dict_uq['objective names'] = dict_uq['objective names'] + labels
        
        samples_plot = samples_plus.copy()
        min_list = dict.fromkeys(samples_plus.columns)
        max_list = dict.fromkeys(samples_plus.columns)
        for i in samples_plus.columns:
            min_temp = min(samples_plot[i])
            min_list[i] = min_temp
            max_temp = max(samples_plot[i])
            max_list[i] = max_temp
            
            if i == self.objective:
                transition_cost_ref = self.get_transition_cost(case_study='ref')
                transition_cost_smr = self.get_transition_cost(case_study='smr')
                nom_values_ref.loc[i] = (transition_cost_ref-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (transition_cost_smr-min_temp)/(max_temp-min_temp)*1-0
            elif i in labels:
                nom_values_ref.loc[i] = (result_ref_full[i]-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (result_smr_full[i]-min_temp)/(max_temp-min_temp)*1-0
                
            samples_plot[i] = (samples_plot[i]-min_temp)/(max_temp-min_temp)*1-0
        
        samples_plot.reset_index(inplace=True)
        samples_plot['Significance'] = 'Neutral'

        dict_uq['draw pdf cdf'] = [False, 1e5]
        
        for i in range(len(output)):
            j = labels[i]
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
            elif out == 'TotalGwp':
                el = 'Total gwp'
            
            output_of_interest = meaning_output[el]
            
            if out == 'F':
                output_of_interest +=' - Capacity'
                output_of_interest += ' [{}; {}] GW'.format(round(min_list[j],1),round(max_list[j],1))
            elif out == 'Ft':
                output_of_interest +=' - Import'
                output_of_interest += ' [{}; {}] TWh'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            elif out == 'TotalGwp':
                output_of_interest += ' [{}; {}] MtCO2'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            
            
            if calc_Sobol:
                dict_uq['objective of interest'] = j
                rheia_uq.run_uq(dict_uq,design_space = 'design_space.csv')
            
            x_meaning = self.uncert_param_meaning.copy()
            
            names, sobol = self.my_post_process_uq.get_sobol(self.case_study, j)
            temp_sobol = [x_meaning[names[m]]+' ('+str(round(100*sobol[m]))+ '%)' for m in range(len(names))]
            dict_sobol = dict.fromkeys(names)
            for k,l in enumerate(names):
                dict_sobol[l] = temp_sobol[k]
            dict_sobol[self.objective] = 'Total transition cost'
            dict_sobol[self.objective] += ' [{}; {}] b€'.format(round(min_list[self.objective]/1000),round(max_list[self.objective]/1000))
            dict_sobol[j] = output_of_interest
            n_threshold = len([i for i in sobol if i > 1/len(sobol)])
            param_to_keep = temp_sobol[:min(n_threshold,6)]
            
            smr_in = False
            for p in param_to_keep:
                if 'SMR' in p:
                    smr_in = True
                    dict_ref_smr = {'Parameter':p,'REF':0}
                    dict_smr_smr = {'Parameter':p,'SMR':0.6}
            
            order_x = [dict_sobol[j]] + param_to_keep + [dict_sobol[self.objective]]
            
            nom_values_plot = nom_values.reset_index()
            nom_values_plot = nom_values_plot.replace({"Parameter": dict_sobol})
            nom_values_plot = nom_values_plot.loc[nom_values_plot['Parameter'].isin(param_to_keep)]
            
            nom_values_ref_plot = nom_values_ref.reset_index()
            nom_values_ref_plot = nom_values_ref_plot.replace({"Parameter": dict_sobol})
            nom_values_ref_plot = nom_values_ref_plot.loc[nom_values_ref_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            
            nom_values_smr_plot = nom_values_smr.reset_index()
            nom_values_smr_plot = nom_values_smr_plot.replace({"Parameter": dict_sobol})
            nom_values_smr_plot = nom_values_smr_plot.loc[nom_values_smr_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            if smr_in:
                nom_values_ref_plot = nom_values_ref_plot.append(dict_ref_smr,ignore_index=True)
                nom_values_smr_plot = nom_values_smr_plot.append(dict_smr_smr,ignore_index=True)

                
                
            
            if focus == 'High':
                share = 20/3
                # temp_high = samples_plot.nlargest(round(len(samples_plot)/(share)),j)
                temp_low = samples_plot.nsmallest(round(len(samples_plot)/(share)),j)
                            
                temp_high = samples_plot.loc[samples_plot[j] >0.1]
                temp_low = samples_plot.loc[samples_plot[j] <0]
                
                # self.get_av_sample(temp_high['Sample'], j+'_'+focus)
                
                s_plot_full = samples_plot.copy()
                
                s_plot_full.drop(labels, axis=1,inplace=True)
                s_plot_full[j] = samples_plot[j]
                
                s_plot_full.loc[s_plot_full['Sample'].isin(temp_high['Sample']),'Significance'] = 'Significant'
                s_plot_full.loc[s_plot_full['Sample'].isin(temp_low['Sample']),'Significance'] = 'Not significant'
                
                s_plot_full = pd.melt(s_plot_full,var_name='x',value_name='value',id_vars=['Significance','Sample'])
                s_plot_full = s_plot_full.replace({"x": dict_sobol})                
                fig = px.strip(s_plot_full,x='x',y='value',color='Significance',
                               color_discrete_map={'Neutral': 'white','Significant':'blue', 'Not significant':'cyan'},
                               stripmode='overlay')
                
                s_plot_sum = s_plot_full.loc[s_plot_full['x'].isin(order_x)]
                
                s_plot_sum.sort_values(by='Significance',inplace=True)
                
                # temp = nom_values_plot.loc[nom_values_plot['Parameter']==output_of_interest,'Nominal'].values[0]
                
                
                if not(flip):
                    fig = px.strip(s_plot_sum,x='x',y='value',color='Significance',
                                    color_discrete_map={'Neutral': 'white','Significant':'blue', 'Not significant':'cyan'},
                                    stripmode='overlay',custom_data=['Sample'])
                    fig.update_layout(xaxis_tickangle=45)
                    fig.add_trace(go.Scatter(x=nom_values_plot["Parameter"], y=nom_values_plot["Nominal"],
                                              mode='markers',
                                              marker=dict(size=15,color='darkorange', symbol='diamond')))
                    fig.add_trace(go.Scatter(x=nom_values_ref_plot["Parameter"], y=nom_values_ref_plot["REF"],
                                              mode='markers',
                                              marker=dict(size=15,color='limegreen', symbol='diamond')))
                    fig.add_trace(go.Scatter(x=nom_values_smr_plot["Parameter"], y=nom_values_smr_plot["SMR"],
                                              mode='markers',
                                              marker=dict(size=15,color='deeppink', symbol='diamond')))
                    
                    xvals=order_x
                    
                    # if temp <= 0.02*max(samples_plot[j]) or temp >= 0.98*max(samples_plot[j]):
                    yvals=[0,1]
                    # else:
                    #     yvals=[0,round(temp,2),1]
                    
                else:
                    fig = px.strip(s_plot_sum,x='value',y='x',color='Significance',
                                    color_discrete_map={'Neutral': 'white','Significant':'blue', 'Not significant':'cyan'},
                                    stripmode='overlay',custom_data=['Sample'])
                    fig.add_trace(go.Scatter(x=nom_values_plot["Nominal"], y=nom_values_plot["Parameter"],
                                              mode='markers',
                                              marker=dict(size=15,color='darkorange', symbol='diamond')))
                    fig.add_trace(go.Scatter(x=nom_values_ref_plot["REF"], y=nom_values_ref_plot["Parameter"],
                                              mode='markers',
                                              marker=dict(size=15,color='limegreen', symbol='diamond')))
                    fig.add_trace(go.Scatter(x=nom_values_smr_plot["SMR"], y=nom_values_smr_plot["Parameter"],
                                              mode='markers',
                                              marker=dict(size=15,color='deeppink', symbol='diamond')))
                    order_x.reverse()
                    
                    # if temp <= 0.02*max(samples_plot[j]) or temp >= 0.98*max(samples_plot[j]):
                    xvals=[0,1]
                    # else:
                    #     xvals=[0,round(temp,2),1]
                        
                    yvals = order_x
                    fig.update_yaxes(categoryorder='array', categoryarray= order_x)
                
                title = None

                self.custom_fig(fig,title,yvals,
                                xvals=xvals,
                                type_graph='strip', flip = flip)
                
                if flip:
                    # if temp <= 0.02*max(samples_plot[j]) or temp >= 0.98*max(samples_plot[j]):
                    xvals = ['Min','Max']
                    fig.update_xaxes(
                        ticktext=xvals,
                        tickvals=[0,1]
                        )
                    # else:
                    #     xvals = ['Min',round(temp,2),'Max']
                    #     fig.update_xaxes(
                    #         ticktext=xvals,
                    #         tickvals=[0,round(temp,2),1]
                    #         )
                else:
                    # if temp <= 0.02*max(samples_plot[j]) or temp >= 0.98*max(samples_plot[j]):
                    yvals = ['Min','Max']
                    fig.update_yaxes(
                        ticktext=yvals,
                        tickvals=list(range(len(yvals)))
                        )
                    # else:
                    #     yvals = ['Min',round(temp,2),'Max']
                    #     fig.update_yaxes(
                    #         ticktext=yvals,
                    #         tickvals=list(range(len(yvals)))
                    #         )
                            
                            
                
                # Define the behavior of the hover tooltip
                fig.update_layout(hovermode='closest')
                if flip:
                    fig.update_traces(hovertemplate='Value: %{x}<br>Sample: %{customdata}')
                else:
                    fig.update_traces(hovertemplate='Value: %{y}<br>Sample: %{customdata}')
                
                pio.show(fig)
                
            if not os.path.exists(Path(self.outdir+"Samples/")):
                Path(self.outdir+"Samples").mkdir(parents=True,exist_ok=True)
            if not os.path.exists(Path(self.outdir+"Samples/_Raw/")):
                Path(self.outdir+"Samples/_Raw").mkdir(parents=True,exist_ok=True)
            
            fig.write_html(self.outdir+"Samples/_Raw/"+j+".html")
            fig.write_image(self.outdir+"Samples/"+j+".pdf", width=1200, height=550)
                
    
                
    
    def get_spec_output_test(self,dict_uq,output,element,year,focus='High',calc_Sobol=False, flip = True):
        
        labels = [None] * len(output)
        samples_plus = self.ampl_uq_collector['Samples'].copy()
        col_objective = samples_plus.columns.get_loc(self.objective)
        samples_plus = samples_plus.iloc[:,:col_objective+1]
        result_ref_full = dict()
        result_smr_full = dict()
        meaning_output = self.dict_meaning()
        for i in range(len(output)):
            
            nom_values = pd.DataFrame(index=samples_plus.columns,columns=['Nominal'],data=0)
            nom_values.index.name='Parameter'
            nom_values.update(self.uncert_nominal)
            
            nom_values_ref = pd.DataFrame(index=samples_plus.columns,columns=['REF'],data=0)
            nom_values_ref.index.name='Parameter'
            
            nom_values_smr = pd.DataFrame(index=samples_plus.columns,columns=['SMR'],data=0)
            nom_values_smr.index.name='Parameter'
            
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
                layer = element[i][1]
            elif out == 'TotalGwp':
                el = element[i]
            y = year[i]
            label = out+'_'+el+'_'+y
            labels[i] = label
            if out == 'F':
                results = self.ampl_uq_collector['Assets'][['F','Sample']]
                results = results.loc[results.index.get_level_values('Technologies') == el]
                results.rename(columns = {out:label},inplace=True)
                
                result_ref = self.ref_results['Assets']['F']
                result_ref = result_ref.loc[result_ref.index.get_level_values('Technologies') == el]
                
                result_smr = self.smr_results['Assets']['F']
                result_smr = result_smr.loc[result_smr.index.get_level_values('Technologies') == el]
                
            elif out == 'Ft':
                results = self.ampl_uq_collector['Year_balance'][[layer,'Sample']]
                results = results.loc[results.index.get_level_values('Elements') == el]
                results.rename(columns = {layer:label},inplace=True)
                
                result_ref = self.ref_results['Year_balance'][layer]
                result_ref = result_ref.loc[result_ref.index.get_level_values('Elements') == el]
                
                result_smr = self.smr_results['Year_balance'][layer]
                result_smr = result_smr.loc[result_smr.index.get_level_values('Elements') == el]
            
            elif out == 'TotalGwp':
                results = self.ampl_uq_collector['TotalGwp'][['TotalGWP','Sample']]
                results.rename(columns = {'TotalGWP':label},inplace=True)
                result_ref = self.ref_results['TotalGwp']['TotalGWP']
                result_smr = self.smr_results['TotalGwp']['TotalGWP']
                results.index.name = 'Years'
                result_ref.index.name = 'Years'
                result_smr.index.name = 'Years'
                
                
            results = results.loc[results.index.get_level_values('Years') == y]
            result_ref = result_ref.loc[result_ref.index.get_level_values('Years') == y]
            result_smr = result_smr.loc[result_smr.index.get_level_values('Years') == y]
            
            if result_ref.empty:
                result_ref_full[label] = 0
            else:
                result_ref_full[label] = result_ref.values[0]
            
            if result_smr.empty:
                result_smr_full[label] = 0
            else:
                result_smr_full[label] = result_smr.values[0]

            results.reset_index(inplace=True)
            results = results.set_index(['Sample'])
            samples_plus[label] = results[label]
            samples_plus.fillna(0,inplace=True)
        
        samples_plus.to_csv(self.samples_file,index=False)
        dict_uq['objective names'] = dict_uq['objective names'] + labels
        
        samples_plot = samples_plus.copy()
        min_list = dict.fromkeys(samples_plus.columns)
        max_list = dict.fromkeys(samples_plus.columns)
        for i in samples_plus.columns:
            min_temp = min(samples_plot[i])
            min_list[i] = min_temp
            max_temp = max(samples_plot[i])
            max_list[i] = max_temp
            
            if i == self.objective:
                transition_cost_ref = self.get_transition_cost(case_study='ref')
                transition_cost_smr = self.get_transition_cost(case_study='smr')
                nom_values_ref.loc[i] = (transition_cost_ref-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (transition_cost_smr-min_temp)/(max_temp-min_temp)*1-0
            elif i in labels:
                nom_values_ref.loc[i] = (result_ref_full[i]-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (result_smr_full[i]-min_temp)/(max_temp-min_temp)*1-0
                
            samples_plot[i] = (samples_plot[i]-min_temp)/(max_temp-min_temp)*1-0
        
        samples_plot.reset_index(inplace=True)
        samples_plot['Significance'] = 'Neutral'

        dict_uq['draw pdf cdf'] = [False, 1e5]
        
        for i in range(len(output)):
            j = labels[i]
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
            elif out == 'TotalGwp':
                el = 'Total gwp'
            
            output_of_interest = meaning_output[el]
            
            if out == 'F':
                output_of_interest +=' - Capacity'
                output_of_interest += ' [{}; {}] GW'.format(round(min_list[j],1),round(max_list[j],1))
            elif out == 'Ft':
                output_of_interest +=' - Import'
                output_of_interest += ' [{}; {}] TWh'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            elif out == 'TotalGwp':
                output_of_interest += ' [{}; {}] MtCO2'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            
            
            if calc_Sobol:
                dict_uq['objective of interest'] = j
                rheia_uq.run_uq(dict_uq,design_space = 'design_space.csv')
            
            x_meaning = self.uncert_param_meaning.copy()
            
            names, sobol = self.my_post_process_uq.get_sobol(self.case_study, j)
            temp_sobol = [x_meaning[names[m]]+' ('+str(round(100*sobol[m]))+ '%)' for m in range(len(names))]
            dict_sobol = dict.fromkeys(names)
            for k,l in enumerate(names):
                dict_sobol[l] = temp_sobol[k]
            dict_sobol[self.objective] = 'Total transition cost'
            dict_sobol[self.objective] += ' [{}; {}] b€'.format(round(min_list[self.objective]/1000),round(max_list[self.objective]/1000))
            dict_sobol[j] = output_of_interest
            n_threshold = len([i for i in sobol if i > 1/len(sobol)])
            param_to_keep = temp_sobol[:min(n_threshold,6)]
            
            smr_in = False
            for p in param_to_keep:
                if 'SMR' in p:
                    smr_in = True
                    dict_ref_smr = {'Parameter':p,'REF':0}
                    dict_smr_smr = {'Parameter':p,'SMR':0.6}
            
            order_x = [dict_sobol[j]] + param_to_keep + [dict_sobol[self.objective]]
            
            nom_values_plot = nom_values.reset_index()
            nom_values_plot = nom_values_plot.replace({"Parameter": dict_sobol})
            nom_values_plot = nom_values_plot.loc[nom_values_plot['Parameter'].isin(param_to_keep)]
            
            nom_values_ref_plot = nom_values_ref.reset_index()
            nom_values_ref_plot = nom_values_ref_plot.replace({"Parameter": dict_sobol})
            nom_values_ref_plot = nom_values_ref_plot.loc[nom_values_ref_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            
            nom_values_smr_plot = nom_values_smr.reset_index()
            nom_values_smr_plot = nom_values_smr_plot.replace({"Parameter": dict_sobol})
            nom_values_smr_plot = nom_values_smr_plot.loc[nom_values_smr_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            if smr_in:
                nom_values_ref_plot = nom_values_ref_plot.append(dict_ref_smr,ignore_index=True)
                nom_values_smr_plot = nom_values_smr_plot.append(dict_smr_smr,ignore_index=True)


                
            s_plot_full = samples_plot.copy()
            
            s_plot_full.drop(labels, axis=1,inplace=True)
            s_plot_full[j] = samples_plot[j]
            
            share = 20/3
            temp_high = samples_plot.nlargest(round(len(samples_plot)/(share)),j)
            temp_low = samples_plot.nsmallest(round(len(samples_plot)/(share)),j)
            
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_high['Sample']),'Significance'] = 'Significant'
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_low['Sample']),'Significance'] = 'Not significant'
            
            for k in s_plot_full.columns:
                if k in dict_sobol.keys():
                    s_plot_full.rename(columns = {k:dict_sobol[k]}, inplace = True)
                elif k == 'Significance':
                    pass
                else:
                    s_plot_full.drop(k, axis=1,inplace=True)
            for k in s_plot_full.columns:
                if not((k in order_x) or (k == 'Significance')):
                    s_plot_full.drop(k, axis=1,inplace=True)
                    
            fig = make_subplots(rows=len(param_to_keep), cols=1,subplot_titles=param_to_keep,
                                shared_xaxes=True)
            
            for k,l in enumerate(param_to_keep):
                fig.add_trace(go.Scatter(x=s_plot_full[dict_sobol[j]], y=s_plot_full[l],mode="markers"),row=k+1, col=1)
            
            fig.update_layout(height=1200, width=550, title_text=dict_sobol[j])
            
            pio.show(fig)
                
            if not os.path.exists(Path(self.outdir+"Samples/TEST/")):
                Path(self.outdir+"Samples/TEST").mkdir(parents=True,exist_ok=True)
            if not os.path.exists(Path(self.outdir+"Samples/TEST/_Raw/")):
                Path(self.outdir+"Samples/TEST/_Raw").mkdir(parents=True,exist_ok=True)
            
            fig.write_html(self.outdir+"Samples/TEST/_Raw/"+j+".html")
            fig.write_image(self.outdir+"Samples/TEST/"+j+".pdf", width=1200, height=550*len(param_to_keep)/2)
    
    def get_spec_output_test_2(self,dict_uq,output,element,year,focus='High',calc_Sobol=False, flip = True):
        
        labels = [None] * len(output)
        samples_plus = self.ampl_uq_collector['Samples'].copy()
        col_objective = samples_plus.columns.get_loc(self.objective)
        samples_plus = samples_plus.iloc[:,:col_objective+1]
        result_ref_full = dict()
        result_smr_full = dict()
        meaning_output = self.dict_meaning()
        for i in range(len(output)):
            
            nom_values = pd.DataFrame(index=samples_plus.columns,columns=['Nominal'],data=0)
            nom_values.index.name='Parameter'
            nom_values.update(self.uncert_nominal)
            
            nom_values_ref = pd.DataFrame(index=samples_plus.columns,columns=['REF'],data=0)
            nom_values_ref.index.name='Parameter'
            
            nom_values_smr = pd.DataFrame(index=samples_plus.columns,columns=['SMR'],data=0)
            nom_values_smr.index.name='Parameter'
            
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
                layer = element[i][1]
            elif out == 'TotalGwp':
                el = element[i]
            y = year[i]
            label = out+'_'+el+'_'+y
            labels[i] = label
            if out == 'F':
                results = self.ampl_uq_collector['Assets'][['F','Sample']]
                results = results.loc[results.index.get_level_values('Technologies') == el]
                results.rename(columns = {out:label},inplace=True)
                
                result_ref = self.ref_results['Assets']['F']
                result_ref = result_ref.loc[result_ref.index.get_level_values('Technologies') == el]
                
                result_smr = self.smr_results['Assets']['F']
                result_smr = result_smr.loc[result_smr.index.get_level_values('Technologies') == el]
                
            elif out == 'Ft':
                results = self.ampl_uq_collector['Year_balance'][[layer,'Sample']]
                results = results.loc[results.index.get_level_values('Elements') == el]
                results.rename(columns = {layer:label},inplace=True)
                
                result_ref = self.ref_results['Year_balance'][layer]
                result_ref = result_ref.loc[result_ref.index.get_level_values('Elements') == el]
                
                result_smr = self.smr_results['Year_balance'][layer]
                result_smr = result_smr.loc[result_smr.index.get_level_values('Elements') == el]
            
            elif out == 'TotalGwp':
                results = self.ampl_uq_collector['TotalGwp'][['TotalGWP','Sample']]
                results.rename(columns = {'TotalGWP':label},inplace=True)
                result_ref = self.ref_results['TotalGwp']['TotalGWP']
                result_smr = self.smr_results['TotalGwp']['TotalGWP']
                results.index.name = 'Years'
                result_ref.index.name = 'Years'
                result_smr.index.name = 'Years'
                
                
            results = results.loc[results.index.get_level_values('Years') == y]
            result_ref = result_ref.loc[result_ref.index.get_level_values('Years') == y]
            result_smr = result_smr.loc[result_smr.index.get_level_values('Years') == y]
            
            if result_ref.empty:
                result_ref_full[label] = 0
            else:
                result_ref_full[label] = result_ref.values[0]
            
            if result_smr.empty:
                result_smr_full[label] = 0
            else:
                result_smr_full[label] = result_smr.values[0]

            results.reset_index(inplace=True)
            results = results.set_index(['Sample'])
            samples_plus[label] = results[label]
            samples_plus.fillna(0,inplace=True)
        
        samples_plus.to_csv(self.samples_file,index=False)
        dict_uq['objective names'] = dict_uq['objective names'] + labels
        
        samples_plot = samples_plus.copy()
        min_list = dict.fromkeys(samples_plus.columns)
        max_list = dict.fromkeys(samples_plus.columns)
        for i in samples_plus.columns:
            min_temp = min(samples_plot[i])
            min_list[i] = min_temp
            max_temp = max(samples_plot[i])
            max_list[i] = max_temp
            
            if i == self.objective:
                transition_cost_ref = self.get_transition_cost(case_study='ref')
                transition_cost_smr = self.get_transition_cost(case_study='smr')
                nom_values_ref.loc[i] = (transition_cost_ref-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (transition_cost_smr-min_temp)/(max_temp-min_temp)*1-0
            elif i in labels:
                nom_values_ref.loc[i] = (result_ref_full[i]-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (result_smr_full[i]-min_temp)/(max_temp-min_temp)*1-0
                
            samples_plot[i] = (samples_plot[i]-min_temp)/(max_temp-min_temp)*1-0
        
        samples_plot.reset_index(inplace=True)
        samples_plot['Significance'] = 'Neutral'

        dict_uq['draw pdf cdf'] = [False, 1e5]
        
        for i in range(len(output)):
            j = labels[i]
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
            elif out == 'TotalGwp':
                el = 'Total gwp'
            
            output_of_interest = meaning_output[el]
            
            if out == 'F':
                output_of_interest +=' - Capacity'
                output_of_interest += ' [{}; {}] GW'.format(round(min_list[j],1),round(max_list[j],1))
            elif out == 'Ft':
                output_of_interest +=' - Import'
                output_of_interest += ' [{}; {}] TWh'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            elif out == 'TotalGwp':
                output_of_interest += ' [{}; {}] MtCO2'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            
            
            if calc_Sobol:
                dict_uq['objective of interest'] = j
                rheia_uq.run_uq(dict_uq,design_space = 'design_space.csv')
            
            x_meaning = self.uncert_param_meaning.copy()
            
            names, sobol = self.my_post_process_uq.get_sobol(self.case_study, j)
            temp_sobol = [x_meaning[names[m]]+' ('+str(round(100*sobol[m]))+ '%)' for m in range(len(names))]
            dict_sobol = dict.fromkeys(names)
            for k,l in enumerate(names):
                dict_sobol[l] = temp_sobol[k]
            dict_sobol[self.objective] = 'Total transition cost'
            dict_sobol[self.objective] += ' [{}; {}] b€'.format(round(min_list[self.objective]/1000),round(max_list[self.objective]/1000))
            dict_sobol[j] = output_of_interest
            n_threshold = len([i for i in sobol if i > 1/len(sobol)])
            param_to_keep = temp_sobol[:min(n_threshold,6)]
            
            smr_in = False
            for p in param_to_keep:
                if 'SMR' in p:
                    smr_in = True
                    dict_ref_smr = {'Parameter':p,'REF':0}
                    dict_smr_smr = {'Parameter':p,'SMR':0.6}
            
            order_x = [dict_sobol[j]] + param_to_keep + [dict_sobol[self.objective]]
            
            nom_values_plot = nom_values.reset_index()
            nom_values_plot = nom_values_plot.replace({"Parameter": dict_sobol})
            nom_values_plot = nom_values_plot.loc[nom_values_plot['Parameter'].isin(param_to_keep)]
            
            nom_values_ref_plot = nom_values_ref.reset_index()
            nom_values_ref_plot = nom_values_ref_plot.replace({"Parameter": dict_sobol})
            nom_values_ref_plot = nom_values_ref_plot.loc[nom_values_ref_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            
            nom_values_smr_plot = nom_values_smr.reset_index()
            nom_values_smr_plot = nom_values_smr_plot.replace({"Parameter": dict_sobol})
            nom_values_smr_plot = nom_values_smr_plot.loc[nom_values_smr_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            if smr_in:
                nom_values_ref_plot = nom_values_ref_plot.append(dict_ref_smr,ignore_index=True)
                nom_values_smr_plot = nom_values_smr_plot.append(dict_smr_smr,ignore_index=True)
                        
                
            s_plot_full = samples_plot.copy()
            
            s_plot_full.drop(labels, axis=1,inplace=True)
            s_plot_full[j] = samples_plot[j]
            
            share = 20/3
            temp_high = samples_plot.nlargest(round(len(samples_plot)/(share)),j)
            temp_low = samples_plot.nsmallest(round(len(samples_plot)/(share)),j)
            
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_high['Sample']),'Significance'] = 'Significant'
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_low['Sample']),'Significance'] = 'Not significant'
            
            s_plot_full = pd.melt(s_plot_full,var_name='x',value_name='value',id_vars=['Significance','Sample',j])
            s_plot_full = s_plot_full.replace({"x": dict_sobol})            
            
            order_x.pop(0)
            order_x.pop()
            
            s_plot_sum = s_plot_full.loc[s_plot_full['x'].isin(order_x)]
            s_plot_sum.rename(columns = {j:dict_sobol[j]}, inplace = True)
            s_plot_sum.sort_values(by='Significance',inplace=True)
            
            
    
            
            
            fig = px.scatter(s_plot_sum,x='value',y=dict_sobol[j],color='x',trendline='rolling',trendline_options=dict(window=int(len(s_plot_sum)/15),center=True,min_periods = 1,win_type='triang'))
            fig.update_traces(visible=False, selector=dict(mode="markers"))
            
            if not os.path.exists(Path(self.outdir+"Samples/TEST_2/")):
                Path(self.outdir+"Samples/TEST_2").mkdir(parents=True,exist_ok=True)
            

            xvals=[0,1]
            yvals=[0,1]
            # # else:
            # #     xvals=[0,round(temp,2),1]
                
            # yvals = order_x
            # fig.update_yaxes(categoryorder='array', categoryarray= order_x)
            
            # A = 4
            
            title = "<b>{}</b><br>Impacting parameters".format(dict_sobol[j])
            title = "<b>{}</b>".format(dict_sobol[j])

            self.custom_fig(fig,title,yvals,
                            xvals=xvals,
                            type_graph='strip', flip = flip)
            
            fig.add_shape(x0=fig.layout.xaxis.tickvals[0],x1=fig.layout.xaxis.tickvals[-1],
                      y0=nom_values_ref.loc[j][0],y1=nom_values_ref.loc[j][0],
                      type='line',layer="above",
                      line=dict(color='rgb(90,90,90)', width=2,dash='dot'),opacity=1)
            
            xvals = ['Min','Max']
            fig.update_xaxes(
                ticktext=xvals,
                tickvals=[0,1]
                )
            
            yvals = [round(min_list[j]/1000,1),round(max_list[j]/1000,1)]
            fig.update_yaxes(
                ticktext=yvals,
                tickvals=[0,1]
                )

            pio.show(fig)
                
            if not os.path.exists(Path(self.outdir+"Samples/TEST_2/")):
                Path(self.outdir+"Samples/TEST_2").mkdir(parents=True,exist_ok=True)
            if not os.path.exists(Path(self.outdir+"Samples/TEST_2/_Raw/")):
                Path(self.outdir+"Samples/TEST_2/_Raw").mkdir(parents=True,exist_ok=True)
            
            fig.write_html(self.outdir+"Samples/TEST_2/_Raw/"+j+".html")
            fig.write_image(self.outdir+"Samples/TEST_2/"+j+".pdf", width=1200, height=550)
     
    
    
    def get_spec_output_test_3(self,dict_uq,output,element,year,focus='High',calc_Sobol=False, flip = True):
        
        labels = [None] * len(output)
        samples_plus = self.ampl_uq_collector['Samples'].copy()
        col_objective = samples_plus.columns.get_loc(self.objective)
        samples_plus = samples_plus.iloc[:,:col_objective+1]
        result_ref_full = dict()
        result_smr_full = dict()
        meaning_output = self.dict_meaning()
        for i in range(len(output)):
            
            nom_values = pd.DataFrame(index=samples_plus.columns,columns=['Nominal'],data=0)
            nom_values.index.name='Parameter'
            nom_values.update(self.uncert_nominal)
            
            nom_values_ref = pd.DataFrame(index=samples_plus.columns,columns=['REF'],data=0)
            nom_values_ref.index.name='Parameter'
            
            nom_values_smr = pd.DataFrame(index=samples_plus.columns,columns=['SMR'],data=0)
            nom_values_smr.index.name='Parameter'
            
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
                layer = element[i][1]
            elif out == 'TotalGwp':
                el = element[i]
            y = year[i]
            label = out+'_'+el+'_'+y
            labels[i] = label
            if out == 'F':
                results = self.ampl_uq_collector['Assets'][['F','Sample']]
                results = results.loc[results.index.get_level_values('Technologies') == el]
                results.rename(columns = {out:label},inplace=True)
                
                result_ref = self.ref_results['Assets']['F']
                result_ref = result_ref.loc[result_ref.index.get_level_values('Technologies') == el]
                
                result_smr = self.smr_results['Assets']['F']
                result_smr = result_smr.loc[result_smr.index.get_level_values('Technologies') == el]
                
            elif out == 'Ft':
                results = self.ampl_uq_collector['Year_balance'][[layer,'Sample']]
                results = results.loc[results.index.get_level_values('Elements') == el]
                results.rename(columns = {layer:label},inplace=True)
                
                result_ref = self.ref_results['Year_balance'][layer]
                result_ref = result_ref.loc[result_ref.index.get_level_values('Elements') == el]
                
                result_smr = self.smr_results['Year_balance'][layer]
                result_smr = result_smr.loc[result_smr.index.get_level_values('Elements') == el]
            
            elif out == 'TotalGwp':
                results = self.ampl_uq_collector['TotalGwp'][['TotalGWP','Sample']]
                results.rename(columns = {'TotalGWP':label},inplace=True)
                result_ref = self.ref_results['TotalGwp']['TotalGWP']
                result_smr = self.smr_results['TotalGwp']['TotalGWP']
                results.index.name = 'Years'
                result_ref.index.name = 'Years'
                result_smr.index.name = 'Years'
                
                
            results = results.loc[results.index.get_level_values('Years') == y]
            result_ref = result_ref.loc[result_ref.index.get_level_values('Years') == y]
            result_smr = result_smr.loc[result_smr.index.get_level_values('Years') == y]
            
            if result_ref.empty:
                result_ref_full[label] = 0
            else:
                result_ref_full[label] = result_ref.values[0]
            
            if result_smr.empty:
                result_smr_full[label] = 0
            else:
                result_smr_full[label] = result_smr.values[0]

            results.reset_index(inplace=True)
            results = results.set_index(['Sample'])
            samples_plus[label] = results[label]
            samples_plus.fillna(0,inplace=True)
        
        samples_plus.to_csv(self.samples_file,index=False)
        dict_uq['objective names'] = dict_uq['objective names'] + labels
        
        samples_plot = samples_plus.copy()
        min_list = dict.fromkeys(samples_plus.columns)
        max_list = dict.fromkeys(samples_plus.columns)
        for i in samples_plus.columns:
            min_temp = min(samples_plot[i])
            min_list[i] = min_temp
            max_temp = max(samples_plot[i])
            max_list[i] = max_temp
            
            if i == self.objective:
                transition_cost_ref = self.get_transition_cost(case_study='ref')
                transition_cost_smr = self.get_transition_cost(case_study='smr')
                nom_values_ref.loc[i] = (transition_cost_ref-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (transition_cost_smr-min_temp)/(max_temp-min_temp)*1-0
            elif i in labels:
                nom_values_ref.loc[i] = (result_ref_full[i]-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (result_smr_full[i]-min_temp)/(max_temp-min_temp)*1-0
                
            samples_plot[i] = (samples_plot[i]-min_temp)/(max_temp-min_temp)*1-0
        
        samples_plot.reset_index(inplace=True)
        samples_plot['Significance'] = 0

        dict_uq['draw pdf cdf'] = [False, 1e5]
        
        for i in range(len(output)):
            j = labels[i]
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
            elif out == 'TotalGwp':
                el = 'Total gwp'
            
            output_of_interest = meaning_output[el]
            
            if out == 'F':
                output_of_interest +=' - Capacity'
                output_of_interest += ' [{}; {}] GW'.format(round(min_list[j],1),round(max_list[j],1))
            elif out == 'Ft':
                output_of_interest +=' - Import'
                output_of_interest += ' [{}; {}] TWh'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            elif out == 'TotalGwp':
                output_of_interest += ' [{}; {}] MtCO2'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            
            
            if calc_Sobol:
                dict_uq['objective of interest'] = j
                rheia_uq.run_uq(dict_uq,design_space = 'design_space.csv')
            
            x_meaning = self.uncert_param_meaning.copy()
            
            names, sobol = self.my_post_process_uq.get_sobol(self.case_study, j)
            temp_sobol = [x_meaning[names[m]]+' ('+str(round(100*sobol[m]))+ '%)' for m in range(len(names))]
            dict_sobol = dict.fromkeys(names)
            for k,l in enumerate(names):
                dict_sobol[l] = temp_sobol[k]
            dict_sobol[self.objective] = 'Total transition cost'
            dict_sobol[self.objective] += ' [{}; {}] b€'.format(round(min_list[self.objective]/1000),round(max_list[self.objective]/1000))
            dict_sobol[j] = output_of_interest
            n_threshold = len([i for i in sobol if i > 1/len(sobol)])
            param_to_keep = temp_sobol[:min(n_threshold,6)]
            param_to_keep = temp_sobol[:6]    

                    
            smr_in = False
            for p in param_to_keep:
                if 'SMR' in p:
                    smr_in = True
                    dict_ref_smr = {'Parameter':p,'REF':0}
                    dict_smr_smr = {'Parameter':p,'SMR':0.6}
            
            order_x = [dict_sobol[j]] + param_to_keep + [dict_sobol[self.objective]]
            
            nom_values_plot = nom_values.reset_index()
            nom_values_plot = nom_values_plot.replace({"Parameter": dict_sobol})
            nom_values_plot = nom_values_plot.loc[nom_values_plot['Parameter'].isin(param_to_keep)]
            
            nom_values_ref_plot = nom_values_ref.reset_index()
            nom_values_ref_plot = nom_values_ref_plot.replace({"Parameter": dict_sobol})
            nom_values_ref_plot = nom_values_ref_plot.loc[nom_values_ref_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            
            nom_values_smr_plot = nom_values_smr.reset_index()
            nom_values_smr_plot = nom_values_smr_plot.replace({"Parameter": dict_sobol})
            nom_values_smr_plot = nom_values_smr_plot.loc[nom_values_smr_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            if smr_in:
                nom_values_ref_plot = nom_values_ref_plot.append(dict_ref_smr,ignore_index=True)
                nom_values_smr_plot = nom_values_smr_plot.append(dict_smr_smr,ignore_index=True)

            share = 20/3
            temp_high = samples_plot.nlargest(round(len(samples_plot)/(share)),j)
            temp_low = samples_plot.nsmallest(round(len(samples_plot)/(share)),j)
            
            # self.get_av_sample(temp_high['Sample'], j+'_'+focus)
            
            s_plot_full = samples_plot.copy()
            
            s_plot_full.drop(labels, axis=1,inplace=True)
            s_plot_full[j] = samples_plot[j]
            
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_high['Sample']),'Significance'] = -1
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_low['Sample']),'Significance'] = 1
            
            s_plot_full = pd.melt(s_plot_full,var_name='x',value_name='value',id_vars=['Significance','Sample'])
            s_plot_full = s_plot_full.replace({"x": dict_sobol})                
            fig = px.strip(s_plot_full,x='x',y='value',color='Significance',
                           color_discrete_map={0: 'white', 1:'blue', -1:'cyan'},
                           stripmode='overlay')
            
            s_plot_sum = s_plot_full.loc[s_plot_full['x'].isin(order_x)]
            
            # s_plot_sum = s_plot_sum.loc[s_plot_sum['Significance'] != 0]
            
            
            n_split = 10
            l = np.linspace(1/n_split,1,n_split)
            mi_temp = pd.MultiIndex.from_product([param_to_keep,l])
            df_av = pd.DataFrame(0,index=mi_temp,columns=['Value'])
            for p in param_to_keep:
                for m,n in enumerate(l):
                    if m==0:
                        inf = 0
                    else:
                        inf = l[m-1]
                    s_plot_sum_temp = s_plot_sum.loc[(s_plot_sum['x']==p) & (s_plot_sum['value'] >= inf) & (s_plot_sum['value'] <= n)]
                    df_av.loc[(p,n),'Value']=np.mean(s_plot_sum_temp['Significance'])
            
            
            df_av.reset_index(inplace=True)
            df_av.level_1=0.1
            
            fig = px.bar(df_av,x='level_1',y='level_0',color='Value',color_continuous_scale='RdBu')
            fig.update_traces(width=0.3)
            # fig.update_coloraxes(showscale=False)
            
            
            order_x.pop()
            order_x.pop(0)
            order_x.reverse()
            
            # if temp <= 0.02*max(samples_plot[j]) or temp >= 0.98*max(samples_plot[j]):
            xvals=[0,1]
            # else:
            #     xvals=[0,round(temp,2),1]
                
            yvals = order_x
            fig.update_yaxes(categoryorder='array', categoryarray= order_x)
            
            A = 4
            
            title = "<b>{}</b><br>Impacting parameters".format(dict_sobol[j])
            title = "<b>{}</b>".format(dict_sobol[j])

            self.custom_fig(fig,title,yvals,
                            xvals=xvals,
                            type_graph='strip', flip = flip)
            
            if flip:
                # if temp <= 0.02*max(samples_plot[j]) or temp >= 0.98*max(samples_plot[j]):
                xvals = ['Min','Max']
                fig.update_xaxes(
                    ticktext=xvals,
                    tickvals=[0,1]
                    )
                # else:
                #     xvals = ['Min',round(temp,2),'Max']
                #     fig.update_xaxes(
                #         ticktext=xvals,
                #         tickvals=[0,round(temp,2),1]
                #         )
            else:
                # if temp <= 0.02*max(samples_plot[j]) or temp >= 0.98*max(samples_plot[j]):
                yvals = ['Min','Max']
                fig.update_yaxes(
                    ticktext=yvals,
                    tickvals=list(range(len(yvals)))
                    )
                # else:
                #     yvals = ['Min',round(temp,2),'Max']
                #     fig.update_yaxes(
                #         ticktext=yvals,
                #         tickvals=list(range(len(yvals)))
                #         )
                        
                        
            
            # Define the behavior of the hover tooltip
            fig.update_layout(hovermode='closest')
            if flip:
                fig.update_traces(hovertemplate='Value: %{x}<br>Sample: %{customdata}')
            else:
                fig.update_traces(hovertemplate='Value: %{y}<br>Sample: %{customdata}')
            
            pio.show(fig)
                
            if not os.path.exists(Path(self.outdir+"Samples/TEST_3/")):
                Path(self.outdir+"Samples/TEST_3").mkdir(parents=True,exist_ok=True)
            if not os.path.exists(Path(self.outdir+"Samples/TEST_3/_Raw/")):
                Path(self.outdir+"Samples/TEST_3/_Raw").mkdir(parents=True,exist_ok=True)
            
            fig.write_html(self.outdir+"Samples/TEST_3/_Raw/"+j+".html")
            fig.write_image(self.outdir+"Samples/TEST_3/"+j+".pdf", width=1200, height=550)
                
    
    def get_spec_output_test_4(self,dict_uq,output,element,year,focus='High',calc_Sobol=False, flip = True):
        
        labels = [None] * len(output)
        samples_plus = self.ampl_uq_collector['Samples'].copy()
        col_objective = samples_plus.columns.get_loc(self.objective)
        samples_plus = samples_plus.iloc[:,:col_objective+1]
        result_ref_full = dict()
        result_smr_full = dict()
        meaning_output = self.dict_meaning()
        for i in range(len(output)):
            
            nom_values = pd.DataFrame(index=samples_plus.columns,columns=['Nominal'],data=0)
            nom_values.index.name='Parameter'
            nom_values.update(self.uncert_nominal)
            
            nom_values_ref = pd.DataFrame(index=samples_plus.columns,columns=['REF'],data=0)
            nom_values_ref.index.name='Parameter'
            
            nom_values_smr = pd.DataFrame(index=samples_plus.columns,columns=['SMR'],data=0)
            nom_values_smr.index.name='Parameter'
            
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
                layer = element[i][1]
            elif out == 'TotalGwp':
                el = element[i]
            y = year[i]
            label = out+'_'+el+'_'+y
            labels[i] = label
            if out == 'F':
                results = self.ampl_uq_collector['Assets'][['F','Sample']]
                results = results.loc[results.index.get_level_values('Technologies') == el]
                results.rename(columns = {out:label},inplace=True)
                
                result_ref = self.ref_results['Assets']['F']
                result_ref = result_ref.loc[result_ref.index.get_level_values('Technologies') == el]
                
                result_smr = self.smr_results['Assets']['F']
                result_smr = result_smr.loc[result_smr.index.get_level_values('Technologies') == el]
                
            elif out == 'Ft':
                results = self.ampl_uq_collector['Year_balance'][[layer,'Sample']]
                results = results.loc[results.index.get_level_values('Elements') == el]
                results.rename(columns = {layer:label},inplace=True)
                
                result_ref = self.ref_results['Year_balance'][layer]
                result_ref = result_ref.loc[result_ref.index.get_level_values('Elements') == el]
                
                result_smr = self.smr_results['Year_balance'][layer]
                result_smr = result_smr.loc[result_smr.index.get_level_values('Elements') == el]
            
            elif out == 'TotalGwp':
                results = self.ampl_uq_collector['TotalGwp'][['TotalGWP','Sample']]
                results.rename(columns = {'TotalGWP':label},inplace=True)
                result_ref = self.ref_results['TotalGwp']['TotalGWP']
                result_smr = self.smr_results['TotalGwp']['TotalGWP']
                results.index.name = 'Years'
                result_ref.index.name = 'Years'
                result_smr.index.name = 'Years'
                
                
            results = results.loc[results.index.get_level_values('Years') == y]
            result_ref = result_ref.loc[result_ref.index.get_level_values('Years') == y]
            result_smr = result_smr.loc[result_smr.index.get_level_values('Years') == y]
            
            if result_ref.empty:
                result_ref_full[label] = 0
            else:
                result_ref_full[label] = result_ref.values[0]
            
            if result_smr.empty:
                result_smr_full[label] = 0
            else:
                result_smr_full[label] = result_smr.values[0]

            results.reset_index(inplace=True)
            results = results.set_index(['Sample'])
            samples_plus[label] = results[label]
            samples_plus.fillna(0,inplace=True)
        
        samples_plus.to_csv(self.samples_file,index=False)
        dict_uq['objective names'] = dict_uq['objective names'] + labels
        
        samples_plot = samples_plus.copy()
        min_list = dict.fromkeys(samples_plus.columns)
        max_list = dict.fromkeys(samples_plus.columns)
        for i in samples_plus.columns:
            min_temp = min(samples_plot[i])
            min_list[i] = min_temp
            max_temp = max(samples_plot[i])
            max_list[i] = max_temp
            
            if i == self.objective:
                transition_cost_ref = self.get_transition_cost(case_study='ref')
                transition_cost_smr = self.get_transition_cost(case_study='smr')
                nom_values_ref.loc[i] = (transition_cost_ref-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (transition_cost_smr-min_temp)/(max_temp-min_temp)*1-0
            elif i in labels:
                nom_values_ref.loc[i] = (result_ref_full[i]-min_temp)/(max_temp-min_temp)*1-0
                nom_values_smr.loc[i] = (result_smr_full[i]-min_temp)/(max_temp-min_temp)*1-0
                
            samples_plot[i] = (samples_plot[i]-min_temp)/(max_temp-min_temp)*1-0
        
        samples_plot.reset_index(inplace=True)
        samples_plot['Significance'] = 0

        dict_uq['draw pdf cdf'] = [False, 1e5]
        
        for i in range(len(output)):
            j = labels[i]
            out = output[i]
            if out == 'F':
                el = element[i]
            elif out == 'Ft':
                el = element[i][0]
            elif out == 'TotalGwp':
                el = 'Total gwp'
            
            output_of_interest = meaning_output[el]
            
            if out == 'F':
                output_of_interest +=' - Capacity'
                output_of_interest += ' [{}; {}] GW'.format(round(min_list[j],1),round(max_list[j],1))
            elif out == 'Ft':
                output_of_interest +=' - Import'
                output_of_interest += ' [{}; {}] TWh'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            elif out == 'TotalGwp':
                output_of_interest += ' [{}; {}] MtCO2'.format(round(min_list[j]/1000,1),round(max_list[j]/1000,1))
            
            
            if calc_Sobol:
                dict_uq['objective of interest'] = j
                rheia_uq.run_uq(dict_uq,design_space = 'design_space.csv')
            
            x_meaning = self.uncert_param_meaning.copy()
            
            names, sobol = self.my_post_process_uq.get_sobol(self.case_study, j)
            temp_sobol = [x_meaning[names[m]]+' ('+str(round(100*sobol[m]))+ '%)' for m in range(len(names))]
            dict_sobol = dict.fromkeys(names)
            for k,l in enumerate(names):
                dict_sobol[l] = temp_sobol[k]
            dict_sobol[self.objective] = 'Total transition cost'
            dict_sobol[self.objective] += ' [{}; {}] b€'.format(round(min_list[self.objective]/1000),round(max_list[self.objective]/1000))
            dict_sobol[j] = output_of_interest
            n_threshold = len([i for i in sobol if i > 1/len(sobol)])
            param_to_keep = temp_sobol[:min(n_threshold,6)]    

                    
            smr_in = False
            for p in param_to_keep:
                if 'SMR' in p:
                    smr_in = True
                    dict_ref_smr = {'Parameter':p,'REF':0}
                    dict_smr_smr = {'Parameter':p,'SMR':0.6}
            
            order_x = [dict_sobol[j]] + param_to_keep + [dict_sobol[self.objective]]
            
            nom_values_plot = nom_values.reset_index()
            nom_values_plot = nom_values_plot.replace({"Parameter": dict_sobol})
            nom_values_plot = nom_values_plot.loc[nom_values_plot['Parameter'].isin(param_to_keep)]
            
            nom_values_ref_plot = nom_values_ref.reset_index()
            nom_values_ref_plot = nom_values_ref_plot.replace({"Parameter": dict_sobol})
            nom_values_ref_plot = nom_values_ref_plot.loc[nom_values_ref_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            
            nom_values_smr_plot = nom_values_smr.reset_index()
            nom_values_smr_plot = nom_values_smr_plot.replace({"Parameter": dict_sobol})
            nom_values_smr_plot = nom_values_smr_plot.loc[nom_values_smr_plot['Parameter'].isin([dict_sobol[j]] + [dict_sobol[self.objective]])]
            if smr_in:
                nom_values_ref_plot = nom_values_ref_plot.append(dict_ref_smr,ignore_index=True)
                nom_values_smr_plot = nom_values_smr_plot.append(dict_smr_smr,ignore_index=True)

            share = 20/3
            temp_high = samples_plot.nlargest(round(len(samples_plot)/(share)),j)
            temp_low = samples_plot.nsmallest(round(len(samples_plot)/(share)),j)
            
            # self.get_av_sample(temp_high['Sample'], j+'_'+focus)
            
            s_plot_full = samples_plot.copy()
            
            s_plot_full.drop(labels, axis=1,inplace=True)
            s_plot_full[j] = samples_plot[j]
            
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_high['Sample']),'Significance'] = -1
            s_plot_full.loc[s_plot_full['Sample'].isin(temp_low['Sample']),'Significance'] = 1
            
            # s_plot_full = pd.melt(s_plot_full,var_name='x',value_name='value',id_vars=['Significance','Sample'])
            # s_plot_full = s_plot_full.replace({"x": dict_sobol})
            s_plot_full.rename(columns=dict_sobol, inplace=True)
            s_plot_sum = s_plot_full[order_x+['Sample']]
            
            order_x.pop()
            order_x.pop(0)
            
            
            fig = go.Figure()
            
            for x in order_x:
                temp = s_plot_sum[[dict_sobol[j],x,'Sample']]
                temp = temp.sort_values(by=x)
                temp_rol = temp[dict_sobol[j]].rolling(window=int(len(temp)/5),center=True,min_periods = 1,win_type='triang').mean()

                fig.add_trace(go.Scatter(x=temp[x],y=temp_rol,line_shape='spline',line=dict(width=2)))

                
                share = 20/3
                temp_high = temp.nlargest(round(len(temp)/(share)),x)
                temp_low = temp.nsmallest(round(len(temp)/(share)),x)
                
                yvals = [0,1]
                xvals = [0,1]
                title = 'Boxplot High - {}'.format(x)
                
                fig_high = px.box(temp_high, y=dict_sobol[j],
                               title=title,notched=True, points='outliers')
                fig_high.update_layout(yaxis_range=yvals)
                
                fig_high.write_image(self.outdir+"Samples/TEST_4/"+j+"_Box_high_{}".format(x)+".pdf", width=1200, height=550)
                fig_high.write_html(self.outdir+"Samples/TEST_4/_Raw/"+j+"_Box_high_{}".format(x)+".html")
                
                title = 'Boxplot Low- {}'.format(x)
                
                fig_low = px.box(temp_low, y=dict_sobol[j],
                               title=title,notched=True, points='outliers')
                fig_low.update_layout(yaxis_range=yvals)
                
                fig_low.write_image(self.outdir+"Samples/TEST_4/"+j+"_Box_low_{}".format(x)+".pdf", width=1200, height=550)
                fig_low.write_html(self.outdir+"Samples/TEST_4/_Raw/"+j+"_Box_low_{}".format(x)+".html")
                

            xvals=[0,1]
            yvals=[0,1]
            # # else:
            # #     xvals=[0,round(temp,2),1]
                
            # yvals = order_x
            # fig.update_yaxes(categoryorder='array', categoryarray= order_x)
            
            # A = 4
            
            title = "<b>{}</b><br>Impacting parameters".format(dict_sobol[j])
            title = "<b>{}</b>".format(dict_sobol[j])

            self.custom_fig(fig,title,yvals,
                            xvals=xvals,
                            type_graph='strip', flip = flip)
            
            fig.add_shape(x0=fig.layout.xaxis.tickvals[0],x1=fig.layout.xaxis.tickvals[-1],
                      y0=nom_values_ref.loc[j][0],y1=nom_values_ref.loc[j][0],
                      type='line',layer="above",
                      line=dict(color='rgb(90,90,90)', width=2,dash='dot'),opacity=1)
            
            xvals = ['Min','Max']
            fig.update_xaxes(
                ticktext=xvals,
                tickvals=[0,1]
                )
            
            yvals = [round(min_list[j]/1000,1),round(result_ref_full[j]/1000,1),round(max_list[j]/1000,1)]
            fig.update_yaxes(
                ticktext=yvals,
                tickvals=[0,nom_values_ref.loc[j][0],1]
                )

            pio.show(fig)
                
            if not os.path.exists(Path(self.outdir+"Samples/TEST_4/")):
                Path(self.outdir+"Samples/TEST_4").mkdir(parents=True,exist_ok=True)
            if not os.path.exists(Path(self.outdir+"Samples/TEST_4/_Raw/")):
                Path(self.outdir+"Samples/TEST_4/_Raw").mkdir(parents=True,exist_ok=True)
            
            fig.write_html(self.outdir+"Samples/TEST_4/_Raw/"+j+".html")
            fig.write_image(self.outdir+"Samples/TEST_4/"+j+".pdf", width=1200, height=550)
        
        
    
    def get_spec_sample(self,sample):
        output_file = os.path.join(Path(self.outdir).parent.absolute(),'Runs/Run{}'.format(sample))
        ampl_0  = self.ampl_obj
        case_study = self.case_study
        ampl_graph = AmplGraph(output_file, ampl_0, case_study)
        ampl_graph.outdir=os.path.join(ampl_graph.outdir,'Run{}/'.format(sample))
        if not os.path.exists(Path(ampl_graph.outdir)):
            Path(ampl_graph.outdir).mkdir(parents=True,exist_ok=True)
        # ampl_graph.graph_resource()
        # ampl_graph.graph_tech_cap()
        ampl_graph.graph_gwp_per_sector()
        ampl_graph.graph_layer()
        ampl_graph.graph_load_factor_scaled()
    
    def get_av_sample(self,sample_list,case):
        uq_collector = self.ampl_uq_collector.copy()
        for key in uq_collector:
            if key != 'Samples':
                temp = uq_collector[key].loc[uq_collector[key]['Sample'].isin(sample_list)]
                if len(temp.index.names) == 1:
                    temp.reset_index(inplace=True)
                    temp = temp.set_index(temp.columns[0])
                temp.drop(['Sample'],axis=1,inplace=True)
                uq_collector[key] = temp.groupby(temp.index.names).mean()
            else:
                uq_collector[key] = uq_collector[key].loc[uq_collector[key].index.get_level_values('Sample').isin(sample_list),:]
        
        pkl_folder = os.path.join(Path(self.outdir).parent.absolute(),'Runs/graphs/{}'.format(case))
        if not os.path.exists(Path(pkl_folder)):
            Path(pkl_folder).mkdir(parents=True,exist_ok=True)
        
        open_file = open(pkl_folder+'/_uq_collector.p',"wb")
        pkl.dump(uq_collector, open_file)
        open_file.close()
        
        output_file = os.path.join(Path(pkl_folder).absolute(),'_uq_collector.p')
        ampl_0  = self.ampl_obj
        case_study = self.case_study
        ampl_graph = AmplGraph(output_file, ampl_0, case_study)
        ampl_graph.outdir=pkl_folder+'/'
        if not os.path.exists(Path(ampl_graph.outdir)):
            Path(ampl_graph.outdir).mkdir(parents=True,exist_ok=True)
        ampl_graph.graph_resource()
        ampl_graph.graph_tech_cap()
        # ampl_graph.graph_gwp_per_sector()
        ampl_graph.graph_layer()
        
        
        
        
    
    def get_transition_cost(self,case_study='ref'):
        ampl_0  = self.ampl_obj
        if case_study == 'ref':
            output_file = self.ref_file
            case_study = self.ref_case
            ampl_graph = AmplGraph(output_file, ampl_0, case_study)
            transition_cost = ampl_graph._compute_transition_cost(self.ref_results)
        elif case_study == 'smr':
            output_file = self.smr_file
            case_study = self.smr_case
            ampl_graph = AmplGraph(output_file, ampl_0, case_study)
            transition_cost = ampl_graph._compute_transition_cost(self.smr_results)
        
        transition_cost_2050 = transition_cost['2050']
        
        return 1000*transition_cost_2050
        
    @staticmethod
    def unpkl(self,case_study = None):
        if case_study == None:
            case_study_dir_path = self.case_study_dir_path
        pkl_file = os.path.join(case_study_dir_path,'_uq_collector.p')
        
        open_file = open(pkl_file,"rb")
        loaded_results = pkl.load(open_file)
        open_file.close()

        return loaded_results
    
    @staticmethod
    def get_nominal(uncert_range):
        uncert_range['Nominal'] = (0-uncert_range['Range_min'])/(uncert_range['Range_max']-uncert_range['Range_min'])*1-0
        if 'f_max_nuclear_smr' in (uncert_range.index):
            uncert_range.loc['f_max_nuclear_smr','Nominal'] = 0
        return pd.DataFrame(uncert_range['Nominal'])
        

    def gather_results(self):
        uq_path = self.case_study_dir_path
        uq_path_runs = uq_path + "/Runs/"
        
        if not(Path(os.path.join(uq_path,'_uq_collector.p')).is_file()):
            uq_collector = {}
            dir = sorted(os.listdir(uq_path_runs))
            for i, file in enumerate(dir):
                if "UQ_scenario" in file:
                    sample = int(file.replace('UQ_scenario_', ''))
                    pickle_path = os.path.join(uq_path_runs, file, '_Results.pkl')
                    open_file = open(pickle_path, "rb")
                    loaded_results = pkl.load(open_file)
                    if not(uq_collector):
                        for key in loaded_results:
                            loaded_results[key]['Sample'] = sample
                        uq_collector = loaded_results
                    else:
                        for key in uq_collector:
                            loaded_results[key]['Sample'] = sample
                            uq_collector[key] = uq_collector[key].append(loaded_results[key])
            
            open_file = open(uq_path+'/_uq_collector.p',"wb")
            pkl.dump(uq_collector, open_file)
            open_file.close()

    def fill_df_sobol_obj(self):
        result_dir = [self.case_study]+self.result_dir
        df_sobol = pd.DataFrame(columns=['Param','Sobol','Ranking','Case'])
        for i, j in enumerate(result_dir):

            names, sobol = self.my_post_process_uq.get_sobol(j, 'cost')
            df_sobol_temp = pd.DataFrame(columns=df_sobol.columns)
            df_sobol_temp.loc[:,'Param'] = names
            df_sobol_temp.loc[:,'Sobol'] = np.array(sobol)
            df_sobol_temp.loc[:,'Ranking'] = list(range(1,len(names)+1))
            df_sobol_temp.loc[:,'Case'] = j
            
            if i!=0:
                df_sobol = pd.concat([df_sobol,df_sobol_temp])
            else:
                df_sobol = df_sobol_temp
            
            self.threshold = min(self.threshold,1/len(names))
        
        self.df_sobol = df_sobol
    
    def fill_df_pdf(self):
        result_dir = [self.case_study]+self.result_dir
        df_pdf = pd.DataFrame(columns=['x_pdf','y_pdf','Case'])

        for i, j in enumerate(result_dir):
            
            x_pdf, y_pdf = self.my_post_process_uq.get_pdf(j, 'cost')
            # Pas de lissage polynomial - garder les données brutes
            # poly_func = self._polyfit_func(x_pdf, y_pdf)
            # y_pdf = poly_func(x_pdf)
            df_pdf_temp = pd.DataFrame(columns=df_pdf.columns)
            df_pdf_temp.loc[:,'x_pdf'] = x_pdf
            df_pdf_temp.loc[:,'y_pdf'] = y_pdf
            df_pdf_temp.loc[:,'Case'] = j 
            
            if i!=0:
                df_pdf = pd.concat([df_pdf,df_pdf_temp])
            else:
                df_pdf = df_pdf_temp
            
        self.df_pdf = df_pdf
    
    def fill_df_cdf(self):
        result_dir = [self.case_study]+self.result_dir
        df_cdf = pd.DataFrame(columns=['x_cdf','y_cdf','Case'])

        for i, j in enumerate(result_dir):
            
            x_cdf, y_cdf = self.my_post_process_uq.get_cdf(j, 'cost')
            df_cdf_temp = pd.DataFrame(columns=df_cdf.columns)
            df_cdf_temp.loc[:,'x_cdf'] = x_cdf
            df_cdf_temp.loc[:,'y_cdf'] = y_cdf
            df_cdf_temp.loc[:,'Case'] = j 
            
            if i!=0:
                df_cdf = pd.concat([df_cdf,df_cdf_temp])
            else:
                df_cdf = df_cdf_temp
            
            self.df_cdf = df_cdf
        
    def filter_df_sobol(self,threshold = 1):
        self.fill_df_sobol_obj()
        threshold_inc = self.threshold*threshold
        param_plot = self.df_sobol.loc[self.df_sobol['Sobol']>=threshold_inc]['Param'].unique()
        self.df_sobol_plot = self.df_sobol.loc[self.df_sobol['Param'].isin(param_plot)]
        
    def graph_sobol(self, threshold=0.0001, decimals=3):
        self.filter_df_sobol(threshold)
        fig = px.bar(self.df_sobol_plot,x='Sobol',y='Param',color='Case',
                      title='Sobol index',orientation='h')
        fig.update_layout(barmode='group', xaxis_tickangle=45)
        pio.show(fig)
        fig.write_html(self.outdir+"Sobol_raw.html")

        temp = self.df_sobol_plot.copy()
        param_meaning = self.uncert_param_meaning.copy()
        param_codes = list(self.df_sobol_plot.Param.unique())
        param_labels = [param_meaning.get(p, p) for p in param_codes]
        x_max = temp['Sobol'].max() if not temp.empty else 1
        if x_max <= 0:
            x_max = 1

        fmt = f"{{:.{int(decimals)}f}}"
        temp['Sobol_label'] = temp['Sobol'].map(lambda v: '0' if round(v, int(decimals)) == 0 else fmt.format(v))
        fig.update_traces(
            text=temp['Sobol_label'],
            textposition='outside',
            cliponaxis=False
        )

        fig.update_layout(
            title=dict(
                text="Sobol index of commissioning time",
                x=0.5,
                xanchor='center'
            ),
            template='simple_white',
            showlegend=False,
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=220, r=100, t=90, b=70),
            yaxis_title=None,
            xaxis=dict(
                visible=False,
                range=[0, x_max*1.05]
            ),
            yaxis=dict(
                showline=True,
                linecolor='rgb(90,90,90)',
                ticks='outside',
                tickcolor='rgb(90,90,90)',
                showgrid=False,
                tickmode='array',
                tickvals=param_codes,
                ticktext=param_labels,
                categoryorder='array',
                categoryarray=param_codes
            )
        )

        fig.write_image(self.outdir+"Sobol.pdf", width=1200, height=550)
        plt.close()

    def graph_loo_evolution(self, objective='cost', plot=True):
        """
        Plot the evolution of the LOO error for PCE orders 1, 2 and 3.

        Parameters
        ----------
        objective : str
            Objective name used in the file name (e.g. 'cost').
        plot : bool
            If True, display the plot in the browser.
        """
        order_to_dirs = {
            1: ['first_order_pce', 'First_order_pce'],
            2: ['second_order_pce', 'Second_order_pce'],
            3: ['third_order_pce', 'Third_order_pce']
        }

        base_uq_dir = Path(self.my_post_process_uq.result_path).absolute()
        loo_values = []

        for order, dir_candidates in order_to_dirs.items():
            loo_value = np.nan
            for dir_name in dir_candidates:
                file_path = base_uq_dir / dir_name / f"full_pce_order_{order}_{objective}.txt"
                if not file_path.exists():
                    continue

                with open(file_path, 'r') as f:
                    file_content = f.read()

                match = re.search(r'^\s*LOO\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', file_content, flags=re.MULTILINE)
                if match is not None:
                    loo_value = float(match.group(1))
                break

            loo_values.append(loo_value)

        df_loo = pd.DataFrame({
            'Order': [1, 2, 3],
            'Model': ['First order', 'Second order', 'Third order'],
            'LOO': loo_values
        })

        if df_loo['LOO'].isna().all():
            raise FileNotFoundError(
                f"No LOO value found in expected PCE files under {base_uq_dir}."
            )

        df_loo_plot = df_loo.dropna(subset=['LOO']).copy()
        x_max = df_loo_plot['LOO'].max() if not df_loo_plot.empty else 1
        if x_max <= 0:
            x_max = 1

        df_loo_plot['LOO_label'] = df_loo_plot['LOO'].map(lambda v: f"{v:.4f}")

        fig = px.bar(
            df_loo_plot,
            x='LOO',
            y='Model',
            orientation='h',
            text='LOO_label',
            title=f"Leave-one-out of PCE"
        )

        fig.update_traces(
            textposition='outside',
            cliponaxis=False,
            marker_color='#1f77b4'
        )
        fig.update_layout(
            template='simple_white',
            showlegend=False,
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=220, r=100, t=90, b=70),
            title=dict(
                text="Leave-one-out of PCE",
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                visible=False,
                range=[0, x_max * 1.05]
            ),
            yaxis=dict(
                title=None,
                showline=True,
                linecolor='rgb(90,90,90)',
                ticks='outside',
                tickcolor='rgb(90,90,90)',
                showgrid=False,
                categoryorder='array',
                categoryarray=['Third order', 'Second order', 'First order']
            )
        )

        if plot:
            pio.show(fig)

        fig.write_html(self.outdir + f"LOO_evolution_{objective}.html")
        fig.write_image(self.outdir + f"LOO_evolution_{objective}.pdf", width=1200, height=550)
        plt.close()
    
    def graph_pdf(self):
        self.fill_df_pdf()
        self.df_pdf['x_pdf'] /= 1e6
        fig = px.line(self.df_pdf,x='x_pdf',y='y_pdf',color='Case',
                      title='PDF per case')
        
        
        fig.write_html(self.outdir+"PDF_raw.html")
        
        title = "<b>PDF of total transition cost</b><br>[10<sup>3</sup>b€]"
        temp = self.df_pdf.copy()
        yvals = [0,max(temp['y_pdf'])]
        
        # Calculer dynamiquement les statistiques à partir des données
        min_cost = temp['x_pdf'].min()
        max_cost = temp['x_pdf'].max()
        mean_cost = temp['x_pdf'].mean()
        median_cost = temp['x_pdf'].median()
        
        # Créer une plage centrée autour des données
        xvals = [round(min_cost, 2), round(median_cost, 2), round(mean_cost, 2), round(max_cost, 2)]
        xvals = sorted(list(set(xvals)))  # Enlever doublons et trier
        
        self.custom_fig(fig,title,yvals,xvals=xvals,flip=True)
        fig.write_image(self.outdir+"PDF.pdf", width=1200, height=550)
        plt.close()
        
    
    def graph_cdf(self):
        self.fill_df_cdf()
        # Convertir en même unité que PDF
        self.df_cdf['x_cdf'] /= 1e6
        fig = px.line(self.df_cdf,x='x_cdf',y='y_cdf',color='Case',
                      title='CDF per case')
        
        fig.write_html(self.outdir+"CDF_raw.html")
        
        title = "<b>CDF of total transition cost</b><br>[10<sup>3</sup>b€]"
        temp = self.df_cdf.copy()
        yvals = [0,0.5,round(max(temp['y_cdf']))]
        
        # Ajouter les valeurs pour l'axe x
        min_cost = temp['x_cdf'].min()
        max_cost = temp['x_cdf'].max()
        median_cost = temp['x_cdf'].median()
        xvals = [round(min_cost, 2), round(median_cost, 2), round(max_cost, 2)]
        xvals = sorted(list(set(xvals)))  # Enlever doublons et trier
        
        self.custom_fig(fig,title,yvals,xvals=xvals,flip=True)
        fig.write_image(self.outdir+"CDF.pdf", width=1200, height=550)
        plt.close()

    def graph_pdf_cdf_monte_carlo(self, ampl_uq_collector=None, plot=True, bins=60):
        """
        PDF/CDF empirique dédiée au cas Monte Carlo à partir des samples de coût.

        Cette méthode ne dépend pas des sorties PCE/Rheia (get_pdf/get_cdf) et
        produit directement les CSV/figures pour le plotting.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        samples = ampl_uq_collector['Samples'].copy()
        if self.objective not in samples.columns:
            raise ValueError(f"Colonne '{self.objective}' introuvable dans 'Samples'.")

        values = pd.to_numeric(samples[self.objective], errors='coerce').dropna().to_numpy(dtype=float)
        if values.size < 2:
            raise ValueError("Pas assez de samples valides pour calculer PDF/CDF Monte Carlo.")

        # Coût en bEUR pour rester cohérent avec les graphes de coût total.
        values_plot = values / 1000.0

        # PDF lissée via KDE gaussienne (plus stable visuellement qu'un histogramme relié).
        n_vals = values_plot.size
        std = float(np.std(values_plot, ddof=1)) if n_vals > 1 else 0.0
        if std <= 0:
            std = max(1e-6, float(np.max(values_plot) - np.min(values_plot)) / 100.0)
        bandwidth = 1.05 * std * (n_vals ** (-1.0 / 5.0))
        bandwidth = max(float(bandwidth), 1e-6)

        x_min = float(np.min(values_plot))
        x_max = float(np.max(values_plot))
        pad = max((x_max - x_min) * 0.08, bandwidth)
        n_grid = int(max(200, min(2000, bins * 8)))
        grid_x = np.linspace(x_min - pad, x_max + pad, n_grid)

        z = (grid_x[:, None] - values_plot[None, :]) / bandwidth
        pdf_y = np.mean(np.exp(-0.5 * z * z) / (bandwidth * np.sqrt(2.0 * np.pi)), axis=1)

        # CDF lissée cohérente avec la PDF: intégrale numérique de la KDE.
        cdf_y = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (pdf_y[1:] + pdf_y[:-1]) * np.diff(grid_x))
        ))
        if cdf_y[-1] > 0:
            cdf_y = cdf_y / cdf_y[-1]

        df_pdf = pd.DataFrame({
            'x_pdf': grid_x,
            'y_pdf': pdf_y,
            'Case': self.case_study
        })
        df_cdf = pd.DataFrame({
            'x_cdf': grid_x,
            'y_cdf': cdf_y,
            'Case': self.case_study
        })

        out_pdf_cdf = self.outdir + "PDF_CDF_MC/"
        out_pdf_cdf_raw = self.outdir + "PDF_CDF_MC/_Raw/"
        if not os.path.exists(Path(out_pdf_cdf)):
            Path(out_pdf_cdf).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(out_pdf_cdf_raw)):
            Path(out_pdf_cdf_raw).mkdir(parents=True, exist_ok=True)

        df_pdf.to_csv(out_pdf_cdf + "PDF_monte_carlo.csv", index=False)
        df_cdf.to_csv(out_pdf_cdf + "CDF_monte_carlo.csv", index=False)

        fig_pdf = px.line(df_pdf, x='x_pdf', y='y_pdf', title='PDF Monte Carlo')
        fig_pdf.write_html(out_pdf_cdf_raw + "PDF_monte_carlo_raw.html")

        title_pdf = "<b>PDF of total transition cost</b>"
        pdf_max = float(df_pdf['y_pdf'].max())
        yvals_pdf = [0.0, round(pdf_max, 2)]
        min_cost = float(df_pdf['x_pdf'].min())
        max_cost = float(df_pdf['x_pdf'].max())
        med_cost = float(np.median(values_plot))
        xvals_pdf = sorted(list(set([round(min_cost, 2), round(med_cost, 2), round(max_cost, 2)])))
        self.custom_fig(fig_pdf, title_pdf, yvals_pdf, xvals=xvals_pdf, flip=True)
        fig_pdf.add_annotation(
            xref='paper',
            yref='paper',
            x=0.995,
            y=0.01,
            text='[b€]',
            showarrow=False,
            xanchor='right',
            yanchor='bottom',
            font=dict(size=18, color='rgb(90,90,90)')
        )
        if fig_pdf.layout.yaxis.tickvals is not None:
            fig_pdf.update_yaxes(
                tickvals=fig_pdf.layout.yaxis.tickvals,
                ticktext=[('0' if float(v) == 0.0 else f"{float(v):.2f}") for v in fig_pdf.layout.yaxis.tickvals]
            )
        if self.ref_case is not None:
            ref_x = 821.7
            ref_y = float(np.interp(ref_x, grid_x, pdf_y))
            fig_pdf.add_trace(go.Scatter(
                x=[ref_x],
                y=[ref_y],
                mode='markers',
                marker=dict(symbol='x', size=10, color='black', line=dict(width=2)),
                showlegend=False,
                hovertemplate=f'Ref: {ref_x:.2f} b€<extra></extra>'
            ))
        fig_pdf.write_image(out_pdf_cdf + "PDF_monte_carlo.pdf", width=1200, height=550)
        if plot:
            pio.show(fig_pdf)
        plt.close()

        fig_cdf = px.line(df_cdf, x='x_cdf', y='y_cdf', title='CDF Monte Carlo')
        fig_cdf.write_html(out_pdf_cdf_raw + "CDF_monte_carlo_raw.html")

        title_cdf = "<b>CDF of total transition cost</b>"
        yvals_cdf = [0, 0.5, 1]
        xvals_cdf = sorted(list(set([round(float(df_cdf['x_cdf'].min()), 2), round(med_cost, 2), round(float(df_cdf['x_cdf'].max()), 2)])))
        self.custom_fig(fig_cdf, title_cdf, yvals_cdf, xvals=xvals_cdf, flip=True)
        fig_cdf.add_annotation(
            xref='paper',
            yref='paper',
            x=0.995,
            y=0.01,
            text='[b€]',
            showarrow=False,
            xanchor='right',
            yanchor='bottom',
            font=dict(size=18, color='rgb(90,90,90)')
        )
        fig_cdf.write_image(out_pdf_cdf + "CDF_monte_carlo.pdf", width=1200, height=550)
        if plot:
            pio.show(fig_cdf)
        plt.close()

        return df_pdf, df_cdf


    def graph_total_cost_scenarios(self, ampl_uq_collector=None, plot=True):
        """
        Compare le coût total des scénarios UQ.

        Le graphique affiche la distribution (boxplot) du coût total
        sur tous les samples UQ.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        samples = ampl_uq_collector['Samples'].copy()
        samples = samples.reset_index()

        cost_col = self.objective
        if cost_col not in samples.columns:
            raise ValueError(f"Colonne '{cost_col}' introuvable dans 'Samples'.")

        cols_to_keep = [cost_col]
        if 'Sample' in samples.columns:
            cols_to_keep = ['Sample'] + cols_to_keep

        results = samples[cols_to_keep].copy()
        results.dropna(how='any', inplace=True)
        results['TotalCost_bEUR'] = results[cost_col] / 1000.0
        results['Scenarios'] = ''

        fig = px.box(
            results,
            x='Scenarios',
            y='TotalCost_bEUR',
            title='Total transition cost by scenario',
            points='outliers',
            color_discrete_sequence=['rgb(90,90,90)']
        )

        # Charger et ajouter la valeur déterministe
        det_total = None
        try:
            det_data, _ = self._load_deterministic_result('Transition_cost')
            if det_data is not None:
                if isinstance(det_data, pd.Series):
                    det_total = float(pd.to_numeric(det_data, errors='coerce').dropna().iloc[0] / 1000.0)
                elif isinstance(det_data, pd.DataFrame):
                    if 'Transition_cost' in det_data.columns:
                        det_series = pd.to_numeric(det_data['Transition_cost'], errors='coerce').dropna()
                        if len(det_series) > 0:
                            det_total = float(det_series.iloc[0] / 1000.0)
                    else:
                        det_numeric = det_data.apply(pd.to_numeric, errors='coerce').stack().dropna()
                        if len(det_numeric) > 0:
                            det_total = float(det_numeric.iloc[0] / 1000.0)

            if det_total is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[''],
                        y=[det_total],
                        mode='markers',
                        name='Deterministic',
                        marker=dict(size=12, color='rgb(90,90,90)', symbol='x'),
                        showlegend=True
                    )
                )
        except Exception:
            pass

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "TotalCost/")):
            Path(self.outdir + "TotalCost/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "TotalCost/_Raw/")):
            Path(self.outdir + "TotalCost/_Raw/").mkdir(parents=True, exist_ok=True)

        fig.write_html(self.outdir + "TotalCost/_Raw/TotalCost_scenarios_raw.html")

        y_min = float(results['TotalCost_bEUR'].min())
        y_max = float(results['TotalCost_bEUR'].max())
        if det_total is not None:
            y_min = min(y_min, float(det_total))
            y_max = max(y_max, float(det_total))
        yvals = [round(y_min, 2), round(y_max, 2)] if y_min != y_max else [round(y_min, 2)]

        title = "<b>Total transition cost across scenarios</b><br>[b€]"
        self.custom_fig(fig, title, yvals, xvals=[''], type_graph='bar')
        fig.update_xaxes(ticks='', ticklen=0)
        fig.write_image(self.outdir + "TotalCost/TotalCost_scenarios.pdf", width=1200, height=550)
        plt.close()

        summary = pd.DataFrame([{
            'count': int(results['TotalCost_bEUR'].count()),
            'mean': float(results['TotalCost_bEUR'].mean()),
            'median': float(results['TotalCost_bEUR'].median()),
            'min': float(results['TotalCost_bEUR'].min()),
            'max': float(results['TotalCost_bEUR'].max())
        }])
        summary.to_csv(self.outdir + "TotalCost/TotalCost_scenarios_summary.csv", index=False)

        return results

    def _format_local_sensitivity_parameter_label(self, parameter_name):
        """Return a readable technology label for local sensitivity parameters."""

        param = str(parameter_name).strip()
        meaning = self.dict_meaning()

        if param in self.uncert_param_meaning:
            return self.uncert_param_meaning[param]

        prefixes = ['cp_', 'c_inv_', 'cost_', 'capex_']
        tech_code = param
        for prefix in prefixes:
            if param.startswith(prefix):
                tech_code = param[len(prefix):]
                break

        if tech_code in meaning:
            return meaning[tech_code]

        pretty = tech_code.replace('_', ' ').strip()
        return pretty if pretty else param

    def graph_local_cost_variation(self, ampl_uq_collector=None, plot=True):
        """
        Plot min/max total cost variation around the deterministic cost for each
        technology in a local sensitivity analysis.

        Expected samples.csv format:
        - first row: deterministic sample
        - then, for each technology cost parameter, two rows: min and max case
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        samples = ampl_uq_collector['Samples'].copy()
        if samples.empty:
            raise ValueError("Le fichier samples.csv est vide.")
        if self.objective not in samples.columns:
            raise ValueError(f"Colonne '{self.objective}' introuvable dans 'Samples'.")

        parameter_cols = [c for c in samples.columns if c != self.objective]
        if len(parameter_cols) == 0:
            raise ValueError("Aucun parametre local n'a ete trouve dans samples.csv.")

        costs = pd.to_numeric(samples[self.objective], errors='coerce')
        if costs.isna().any():
            raise ValueError("Certaines valeurs de cout dans samples.csv sont invalides.")

        deterministic_cost = float(costs.iloc[0])
        expected_rows = 1 + 2 * len(parameter_cols)
        if len(samples) < expected_rows:
            raise ValueError(
                "Format Local sensitivity invalide: attendu au moins "
                f"{expected_rows} lignes pour {len(parameter_cols)} parametres, "
                f"mais {len(samples)} lignes ont ete trouvees."
            )

        records = []
        for idx, parameter in enumerate(parameter_cols):
            row_min = 1 + 2 * idx
            row_max = row_min + 1

            cost_min = float(costs.iloc[row_min])
            cost_max = float(costs.iloc[row_max])

            delta_min = cost_min - deterministic_cost
            delta_max = cost_max - deterministic_cost

            delta_mn = min(delta_min, delta_max)
            delta_mx = max(delta_min, delta_max)
            records.append({
                'parameter': parameter,
                'technology': self._format_local_sensitivity_parameter_label(parameter),
                'deterministic_cost_MEUR': deterministic_cost,
                'min_cost_MEUR': min(cost_min, cost_max),
                'max_cost_MEUR': max(cost_min, cost_max),
                'delta_min_MEUR': delta_mn,
                'delta_max_MEUR': delta_mx,
                'delta_min_bEUR': delta_mn / 1000.0,
                'delta_max_bEUR': delta_mx / 1000.0,
                'delta_min_pct': 100.0 * delta_mn / deterministic_cost if deterministic_cost != 0 else np.nan,
                'delta_max_pct': 100.0 * delta_mx / deterministic_cost if deterministic_cost != 0 else np.nan,
                'sample_min': row_min + 1,
                'sample_max': row_max + 1,
            })

        summary = pd.DataFrame(records)
        if summary.empty:
            raise ValueError("Aucune variation locale de cout n'a pu etre construite.")

        order_labels = summary['technology'].tolist()[::-1]

        fig = go.Figure()
        line_color = 'rgb(90,90,90)'
        min_color = 'rgb(150,150,150)'
        max_color = 'rgb(30,30,30)'

        for _, row in summary.iterrows():
            tech_label = row['technology']
            fig.add_trace(
                go.Scatter(
                    x=[row['delta_min_bEUR'], row['delta_max_bEUR']],
                    y=[tech_label, tech_label],
                    mode='lines',
                    line=dict(color=line_color, width=4),
                    hoverinfo='skip',
                    showlegend=False
                )
            )
            x_min_val = row['delta_min_bEUR']
            text_min_pos = 'middle left' if x_min_val <= 0 else 'middle right'
            fig.add_trace(
                go.Scatter(
                    x=[x_min_val],
                    y=[tech_label],
                    mode='markers+text',
                    marker=dict(color=min_color, size=11),
                    text=[f"{round(x_min_val, 1)}"],
                    textposition=text_min_pos,
                    textfont=dict(size=14, color=min_color),
                    customdata=[[row['parameter'], row['sample_min'], row['delta_min_pct']]],
                    hovertemplate=(
                        'Technology: %{y}<br>'
                        'Parameter: %{customdata[0]}<br>'
                        'Min variation: %{x:.2f} b€<br>'
                        'Min variation: %{customdata[2]:.3f}%<br>'
                        'Sample: %{customdata[1]}<extra></extra>'
                    ),
                    showlegend=False
                )
            )
            x_max_val = row['delta_max_bEUR']
            text_max_pos = 'middle right' if x_max_val >= 0 else 'middle left'
            fig.add_trace(
                go.Scatter(
                    x=[x_max_val],
                    y=[tech_label],
                    mode='markers+text',
                    marker=dict(color=max_color, size=11),
                    text=[f"{round(x_max_val, 1)}"],
                    textposition=text_max_pos,
                    textfont=dict(size=14, color=max_color),
                    customdata=[[row['parameter'], row['sample_max'], row['delta_max_pct']]],
                    hovertemplate=(
                        'Technology: %{y}<br>'
                        'Parameter: %{customdata[0]}<br>'
                        'Max variation: %{x:.2f} b€<br>'
                        'Max variation: %{customdata[2]:.3f}%<br>'
                        'Sample: %{customdata[1]}<extra></extra>'
                    ),
                    showlegend=False
                )
            )

        fig.add_shape(
            type='line',
            x0=0,
            x1=0,
            y0=-0.5,
            y1=len(order_labels) - 0.5,
            line=dict(color='rgb(90,90,90)', width=2, dash='dot'),
            layer='below'
        )

        det_cost_bEUR = round(deterministic_cost / 1000.0, 1)
        fig.update_yaxes(categoryorder='array', categoryarray=order_labels)
        fig.update_layout(
            title=f'Local sensitivity - Total cost variation around deterministic scenario of {det_cost_bEUR} [b€]',
            hovermode='closest'
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + 'LocalSensitivity/'
        out_dir_raw = out_dir + '_Raw/'
        if not os.path.exists(Path(out_dir)):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(out_dir_raw)):
            Path(out_dir_raw).mkdir(parents=True, exist_ok=True)

        fig.write_html(out_dir_raw + 'Local_cost_variation_raw.html')

        x_min = float(summary['delta_min_bEUR'].min())
        x_max = float(summary['delta_max_bEUR'].max())
        if np.isclose(x_min, x_max):
            xvals = [round(x_min, 1)]
        else:
            xvals = sorted(set([round(x_min, 1), 0.0, round(x_max, 1)]))

        title = f'Local sensitivity - Total cost variation around deterministic scenario of {det_cost_bEUR} [b€]'
        self.custom_fig(
            fig,
            title,
            order_labels,
            xvals=xvals,
            x_unit='[b€]',
            type_graph='bar',
            neg_value=True,
            flip=True
        )
        # Increase bottom margin so the [b€] annotation (placed at y=-0.10 by custom_fig)
        # is fully within the kaleido rendered canvas and not clipped on PDF export.
        fig.update_layout(margin_b=80)

        fig.write_image(out_dir + 'Local_cost_variation.pdf', width=1200, height=550)

        fig_export = go.Figure(fig)
        fig_export.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.write_image(out_dir + 'Local_cost_variation.png', width=1200, height=550)
        plt.close()

        summary.to_csv(out_dir + 'Local_cost_variation_summary.csv', index=False)

        return summary
    
    def graph_total_gwp_variation(self, ampl_uq_collector=None, plot=True):
        """
        Compare le GWP total cumulé des scénarios UQ.

        Le graphique affiche la distribution (boxplot) du GWP total cumulé
        (somme de TotalGWP sur toutes les années) pour chaque sample UQ.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'TotalGwp' not in ampl_uq_collector:
            raise ValueError("'TotalGwp' introuvable dans le collecteur UQ.")

        total_gwp = ampl_uq_collector['TotalGwp'].copy().reset_index()
        if 'TotalGWP' not in total_gwp.columns:
            raise ValueError("Colonne 'TotalGWP' introuvable dans 'TotalGwp'.")

        sample_col = 'Sample' if 'Sample' in total_gwp.columns else None
        if sample_col is None:
            total_gwp['Sample'] = 1
            sample_col = 'Sample'

        total_gwp['TotalGWP'] = pd.to_numeric(total_gwp['TotalGWP'], errors='coerce')
        total_gwp.dropna(subset=['TotalGWP'], inplace=True)

        results = (
            total_gwp
            .groupby(sample_col, as_index=False)['TotalGWP']
            .sum()
            .rename(columns={'TotalGWP': 'TotalGWP_cumulated_ktCO2'})
        )

        # Keep a readable unit for plotting.
        results['TotalGWP_cumulated_MtCO2'] = results['TotalGWP_cumulated_ktCO2'] / 1000.0
        results['Scenarios'] = ''

        fig = px.box(
            results,
            x='Scenarios',
            y='TotalGWP_cumulated_MtCO2',
            title='Cumulated total GWP by scenario',
            points='outliers',
            color_discrete_sequence=['rgb(90,90,90)']
        )

        # Add markers for scenarios with min/max total transition cost.
        try:
            if 'Samples' in ampl_uq_collector and sample_col in results.columns:
                samples_cost = ampl_uq_collector['Samples'].copy().reset_index()
                cost_col = self.objective if self.objective in samples_cost.columns else None
                if cost_col is not None:
                    if sample_col in samples_cost.columns:
                        sample_cost_col = sample_col
                    elif 'Sample' in samples_cost.columns:
                        sample_cost_col = 'Sample'
                    elif 'index' in samples_cost.columns:
                        sample_cost_col = 'index'
                    else:
                        sample_cost_col = None

                    if sample_cost_col is not None:
                        samples_cost[cost_col] = pd.to_numeric(samples_cost[cost_col], errors='coerce')
                        samples_cost = samples_cost.dropna(subset=[cost_col, sample_cost_col])

                        if not samples_cost.empty:
                            min_cost_idx = samples_cost[cost_col].idxmin()
                            max_cost_idx = samples_cost[cost_col].idxmax()
                            min_cost_sample = samples_cost.loc[min_cost_idx, sample_cost_col]
                            max_cost_sample = samples_cost.loc[max_cost_idx, sample_cost_col]

                            def _add_cost_marker(sample_id, label, color):
                                match = results.loc[
                                    results[sample_col] == sample_id,
                                    'TotalGWP_cumulated_MtCO2'
                                ]
                                if not match.empty:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[''],
                                            y=[float(match.iloc[0])],
                                            mode='markers',
                                            name=label,
                                            marker=dict(size=12, color=color, symbol='x'),
                                            showlegend=True
                                        )
                                    )

                            if min_cost_sample == max_cost_sample:
                                _add_cost_marker(min_cost_sample, 'Min/Max cout transition', 'rgb(80,80,80)')
                            else:
                                _add_cost_marker(min_cost_sample, 'Min cout transition', 'rgb(0,140,0)')
                                _add_cost_marker(max_cost_sample, 'Max cout transition', 'rgb(200,0,0)')
        except Exception:
            pass

        # Add deterministic marker if available.
        try:
            det_data, _ = self._load_deterministic_result('TotalGwp')
            if det_data is not None and 'TotalGWP' in det_data.columns:
                det_series = pd.to_numeric(det_data['TotalGWP'], errors='coerce').dropna()
                if len(det_series) > 0:
                    det_total = float(det_series.sum() / 1000.0)
                    fig.add_trace(
                        go.Scatter(
                            x=[''],
                            y=[det_total],
                            mode='markers',
                            name='Deterministic',
                            marker=dict(size=12, color='rgb(90,90,90)', symbol='x'),
                            showlegend=True
                        )
                    )
                    fig.add_trace(
                        go.Violin(
                            y=[det_total],
                            name='Deterministic',
                            line=dict(color='rgb(90,90,90)'),
                            showlegend=True
                        )
                    )
        except Exception:
            pass

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "TotalGWP/")):
            Path(self.outdir + "TotalGWP/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "TotalGWP/_Raw/")):
            Path(self.outdir + "TotalGWP/_Raw/").mkdir(parents=True, exist_ok=True)

        fig.write_html(self.outdir + "TotalGWP/_Raw/TotalGWP_total_scenarios_raw.html")

        y_min = float(results['TotalGWP_cumulated_MtCO2'].min())
        y_max = float(results['TotalGWP_cumulated_MtCO2'].max())
        yvals = [round(y_min, 2), round(y_max, 2)] if y_min != y_max else [round(y_min, 2)]

        title = "<b>Cumulated total GWP across scenarios</b><br>[MtCO2-eq]"
        self.custom_fig(fig, title, yvals, xvals=[''], type_graph='bar')
        fig.update_xaxes(ticks='', ticklen=0)
        fig.write_image(self.outdir + "TotalGWP/TotalGWP_total_scenarios.pdf", width=1200, height=550)
        plt.close()

        summary = pd.DataFrame([{
            'count': int(results['TotalGWP_cumulated_MtCO2'].count()),
            'mean': float(results['TotalGWP_cumulated_MtCO2'].mean()),
            'median': float(results['TotalGWP_cumulated_MtCO2'].median()),
            'min': float(results['TotalGWP_cumulated_MtCO2'].min()),
            'max': float(results['TotalGWP_cumulated_MtCO2'].max())
        }])
        summary.to_csv(self.outdir + "TotalGWP/TotalGWP_total_scenarios_summary.csv", index=False)

        return results

    def graph_total_gwp_per_year_scenarios(self, ampl_uq_collector=None, plot=True,
                                           exclude_years=None, exclude_2020=False,
                                           exclude_2021=False,
                                           show_min_max_deterministic=False, show_det=False):
        """
        Compare le Total GWP des scénarios UQ par année.

        Le graphique affiche la distribution (violin plot) du TotalGWP
        sur tous les samples pour chaque année, avec trajectoires pour min/déterministe/max.

        Parameters
        ----------
        exclude_years : list[str|int] | None
            Liste optionnelle d'années à exclure (ex: [2020, 2021]).
        exclude_2020 : bool
            Raccourci pour exclure l'année 2020.
        exclude_2021 : bool
            Raccourci pour exclure l'année 2021.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'TotalGwp' not in ampl_uq_collector:
            raise ValueError("'TotalGwp' introuvable dans le collecteur UQ.")

        results = ampl_uq_collector['TotalGwp'].copy()
        results = results.reset_index()

        if 'TotalGWP' not in results.columns:
            raise ValueError("Colonne 'TotalGWP' introuvable dans 'TotalGwp'.")

        year_col = 'Years' if 'Years' in results.columns else results.columns[0]
        sample_col = 'Sample' if 'Sample' in results.columns else None

        line_data = None
        if sample_col is not None:
            line_data = results[[sample_col, year_col, 'TotalGWP']].copy()

        results = results[[year_col, 'TotalGWP']].copy()
        results.dropna(how='any', inplace=True)

        results[year_col] = results[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).replace('YEAR_', '') for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')
        if exclude_2021:
            years_to_exclude.add('2021')

        if years_to_exclude:
            results = results.loc[~results[year_col].isin(years_to_exclude)].copy()

        if results.empty:
            raise ValueError("Aucune donnée annuelle restante après exclusion des années demandées.")

        results['TotalGWP_MtCO2'] = results['TotalGWP'] / 1000.0

        def _year_sort_key(value):
            nums = re.findall(r'\d+', str(value))
            return int(nums[0]) if len(nums) > 0 else 10**9

        ordered_years = sorted(results[year_col].unique().tolist(), key=_year_sort_key)

        # ── Violin plot : une trace par année ──────────────────────────────────────
        fig = go.Figure()

        for year in ordered_years:
            year_vals = results.loc[results[year_col] == year, 'TotalGWP_MtCO2']
            fig.add_trace(
                go.Violin(
                    x=[year] * len(year_vals),
                    y=year_vals,
                    name=year,
                    showlegend=False,
                    #box_visible=True,
                    box_visible=False,
                    meanline_visible=False,
                    fillcolor='rgba(90,90,90,0.25)',
                    line_color='rgb(90,90,90)',
                    #points='outliers',
                    points=False,
                    marker=dict(
                        color='rgb(90,90,90)',
                        size=4,
                        opacity=0.5,
                    ),
                    width=0.8,
                )
            )

        # ── Trajectoires Min / Déterministe / Max (optionnel) ─────────────────────
        if show_min_max_deterministic and line_data is not None and not line_data.empty:
            line_data.dropna(how='any', inplace=True)
            line_data[year_col] = line_data[year_col].astype(str).str.replace('YEAR_', '', regex=False)
            if years_to_exclude:
                line_data = line_data.loc[~line_data[year_col].isin(years_to_exclude)].copy()
            if line_data.empty:
                line_data = None

        if show_min_max_deterministic and line_data is not None and not line_data.empty:
            line_data['TotalGWP_MtCO2'] = line_data['TotalGWP'] / 1000.0

            sample_total = line_data.groupby(sample_col)['TotalGWP_MtCO2'].sum()
            if not sample_total.empty:
                min_sample, max_sample, samples_params = self._get_min_max_samples_from_cost(ampl_uq_collector)

                if min_sample is None or max_sample is None:
                    min_sample = sample_total.idxmin()
                    max_sample = sample_total.idxmax()

                print("Scenarios used for Total GWP trajectories:")
                print(f"- Min (UQ): sample={min_sample}")
                print(f"- Max (UQ): sample={max_sample}")

                if isinstance(samples_params, pd.DataFrame) and not samples_params.empty:
                    params_df = samples_params.copy().reset_index()
                    cp_cols = [c for c in params_df.columns if str(c).startswith('cp_')]

                    if cp_cols:
                        print("Construction times (exp, rounded to 1 decimal):")
                        print(",".join(cp_cols))

                        for label, sid in [('Min', min_sample), ('Max', max_sample)]:
                            sid_num = pd.to_numeric(pd.Series([sid]), errors='coerce').iloc[0]
                            if pd.isna(sid_num):
                                print(f"{label}: sample={sid} -> invalid sample id")
                                continue

                            row_pos = int(round(float(sid_num))) - 1
                            if row_pos < 0 or row_pos >= len(params_df):
                                print(f"{label}: sample={sid} -> row out of bounds in Samples")
                                continue

                            row = params_df.iloc[[row_pos]]
                            match_mode = 'sample->row(sid-1)'
                            if row.empty:
                                print(f"{label}: sample={sid} -> no matching row found in Samples")
                                continue

                            row_index = int(row.index[0])
                            csv_line = row_index + 2
                            print(
                                f"{label}: sample={sid} -> row_index={row_index}, csv_line={csv_line} (match={match_mode})"
                            )

                            vals = np.exp(pd.to_numeric(row.iloc[0][cp_cols], errors='coerce')).round(1)
                            vals_txt = ",".join([str(v) for v in vals.tolist()])
                            print(f"{label},{vals_txt}")

                det_data, det_pkl_path = self._load_deterministic_result('TotalGwp')

                if det_data is not None:
                    print(f"- Deterministic: loaded from {det_pkl_path}")
                else:
                    print(f"- Deterministic: unavailable at {det_pkl_path}")

                scenario_specs = [
                    ('Min', min_sample, 'rgb(0,140,0)', None),
                    ('Deterministic', None, 'rgb(100,100,100)', det_data),
                    ('Max', max_sample, 'rgb(200,0,0)', None)
                ]

                for label, sid in [('Min', min_sample), ('Max', max_sample)]:
                    _, traj_mode = self._match_rows_by_sample_strict(line_data, sample_col, sid)
                    print(f"- {label} trajectory match in TotalGwp: {traj_mode}")

                for label, sample_id, color, det_df in scenario_specs:
                    if label == 'Deterministic' and det_df is not None:
                        scenario_df = det_df.reset_index()
                        det_year_col = 'Years' if 'Years' in scenario_df.columns else scenario_df.columns[0]
                        if 'TotalGWP' in scenario_df.columns:
                            det_gwp_col = 'TotalGWP'
                        elif 'TotalGwp' in scenario_df.columns:
                            det_gwp_col = 'TotalGwp'
                        else:
                            candidate_cols = [c for c in scenario_df.columns if c != det_year_col]
                            det_gwp_col = candidate_cols[-1]

                        scenario_df = scenario_df[[det_year_col, det_gwp_col]].copy()
                        scenario_df.rename(columns={det_year_col: year_col, det_gwp_col: 'TotalGWP'}, inplace=True)
                        scenario_df[year_col] = scenario_df[year_col].astype(str).str.replace('YEAR_', '', regex=False)
                        if years_to_exclude:
                            scenario_df = scenario_df.loc[~scenario_df[year_col].isin(years_to_exclude)].copy()
                        if scenario_df.empty:
                            continue
                        scenario_df['TotalGWP_MtCO2'] = scenario_df['TotalGWP'] / 1000.0
                        scenario_df['__order'] = scenario_df[year_col].map(_year_sort_key)
                        scenario_df.sort_values(by='__order', inplace=True)
                    else:
                        scenario_df, _ = self._match_rows_by_sample_strict(line_data, sample_col, sample_id)
                        scenario_df = scenario_df.copy()
                        if scenario_df.empty:
                            continue
                        scenario_df['__order'] = scenario_df[year_col].map(_year_sort_key)
                        scenario_df.sort_values(by='__order', inplace=True)

                    fig.add_trace(
                        go.Scatter(
                            x=scenario_df[year_col],
                            y=scenario_df['TotalGWP_MtCO2'],
                            #mode='lines+markers',
                            mode='markers',
                            name=label,
                            line=dict(color=color, width=2.5),
                            marker=dict(size=7, color=color, line=dict(color='white', width=1)),
                            legendgroup=label,
                        )
                    )

        # ── DET / NCT trajectoires légères ────────────────────────────────────────
        if show_det:
            det_paths = {
                'DET': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/Deterministic (mean value)/_Results.pkl',
                'NCT': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/No commissioning/_Results.pkl',
            }
            det_colors = {'DET': 'rgba(60,60,60,0.5)', 'NCT': 'rgba(180,60,60,0.5)'}
            for label, path in det_paths.items():
                try:
                    with open(path, 'rb') as f:
                        det_res = pkl.load(f)
                    gwp_key = 'TotalGwp' if 'TotalGwp' in det_res else ('TotalGWP' if 'TotalGWP' in det_res else None)
                    if gwp_key is None:
                        continue
                    det_df = det_res[gwp_key].copy().reset_index()
                    det_year_col = 'Years' if 'Years' in det_df.columns else det_df.columns[0]
                    gwp_val_col = ('TotalGWP' if 'TotalGWP' in det_df.columns
                                   else 'TotalGwp' if 'TotalGwp' in det_df.columns
                                   else det_df.columns[-1])
                    det_df[det_year_col] = det_df[det_year_col].astype(str).str.replace('YEAR_', '', regex=False)
                    if years_to_exclude:
                        det_df = det_df[~det_df[det_year_col].isin(years_to_exclude)]
                    det_df['TotalGWP_MtCO2'] = det_df[gwp_val_col] / 1000.0
                    det_df['__order'] = det_df[det_year_col].map(_year_sort_key)
                    det_df.sort_values('__order', inplace=True)
                    fig.add_trace(go.Scatter(
                        x=det_df[det_year_col], y=det_df['TotalGWP_MtCO2'],
                        mode='lines+markers',
                        line=dict(color=det_colors[label], width=1.5),
                        marker=dict(size=4, color=det_colors[label]),
                        showlegend=False,
                    ))
                except Exception:
                    pass

        fig.update_layout(
            title='Total GWP by scenario',
            violinmode='overlay',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years),
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "TotalGWP/")):
            Path(self.outdir + "TotalGWP/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "TotalGWP/_Raw/")):
            Path(self.outdir + "TotalGWP/_Raw/").mkdir(parents=True, exist_ok=True)

        fig.write_html(self.outdir + "TotalGWP/_Raw/TotalGWP_scenarios_raw.html")

        y_min = float(results['TotalGWP_MtCO2'].min())
        y_max = float(results['TotalGWP_MtCO2'].max())
        yvals = [round(y_min, 1), round(y_max, 1)] if y_min != y_max else [round(y_min, 1)]

        title = "<b>Total GWP across scenarios</b><br>[MtCO2-eq]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        first_year = ordered_years[0] if ordered_years else None
        fig.update_xaxes(
            tickvals=ordered_years,
            ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
        )
        fig.update_layout(
            showlegend=bool(show_min_max_deterministic),
            legend=dict(
                x=0.99,
                y=0.99,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(90,90,90,0.3)',
                borderwidth=1
            )
        )
        fig.write_image(self.outdir + "TotalGWP/TotalGWP_scenarios.pdf", width=1200, height=550)
        plt.close()

        summary = (
            results
            .groupby(year_col, as_index=False)['TotalGWP_MtCO2']
            .agg(['count', 'mean', 'median', 'min', 'max'])
            .reset_index()
            .rename(columns={year_col: 'Year'})
        )
        summary.to_csv(self.outdir + "TotalGWP/TotalGWP_scenarios_summary.csv", index=False)

        return results
    
    def graph_total_cost_per_year_scenarios(self, ampl_uq_collector=None, plot=True,
                                            exclude_years=None, exclude_2020=False,
                                            show_min_max_deterministic=False,
                                            show_det=False):
        """
        Compare le coût total annuel des scénarios UQ par année.

        Le graphique affiche la distribution (violin plot) du coût système annuel
        sur tous les samples UQ, pour chaque année.

        Parameters
        ----------
        exclude_years : list[str|int] | None
            Liste optionnelle d'années à exclure (ex: [2020, '2030']).
        exclude_2020 : bool
            Raccourci pour exclure l'année 2020.
        show_min_max_deterministic : bool
            Si True, ajoute les trajectoires Min / Déterministe / Max au-dessus
            des violons. Si False, affiche uniquement la distribution annuelle.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'TotalCost' not in ampl_uq_collector:
            raise ValueError("'TotalCost' introuvable dans le collecteur UQ.")

        results = ampl_uq_collector['TotalCost'].copy().reset_index()

        year_col = 'Years' if 'Years' in results.columns else ('index' if 'index' in results.columns else results.columns[0])
        sample_col = 'Sample' if 'Sample' in results.columns else None
        if 'TotalCost' not in results.columns:
            raise ValueError("Colonne 'TotalCost' introuvable dans 'TotalCost'.")

        line_data = None
        if sample_col is not None:
            line_data = results[[sample_col, year_col, 'TotalCost']].copy()

        results = results[[year_col, 'TotalCost']].copy()
        results.dropna(how='any', inplace=True)
        results[year_col] = results[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).replace('YEAR_', '') for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')

        if years_to_exclude:
            results = results.loc[~results[year_col].isin(years_to_exclude)].copy()

        if results.empty:
            raise ValueError("Aucune donnée annuelle restante après exclusion des années demandées.")

        results['TotalCost_bEUR'] = results['TotalCost'] / 1000.0

        def _year_sort_key(value):
            nums = re.findall(r'\d+', str(value))
            return int(nums[0]) if len(nums) > 0 else 10**9

        ordered_years = sorted(results[year_col].unique().tolist(), key=_year_sort_key)

        # ── Violin plot : une trace par année ──────────────────────────────────────
        fig = go.Figure()

        for year in ordered_years:
            year_vals = results.loc[results[year_col] == year, 'TotalCost_bEUR']
            fig.add_trace(
                go.Violin(
                    x=[year] * len(year_vals),
                    y=year_vals,
                    name=year,
                    showlegend=False,
                    #box_visible=True,
                    box_visible=False,
                    meanline_visible=False,
                    fillcolor='rgba(90,90,90,0.25)',
                    line_color='rgb(90,90,90)',
                    #points='outliers',
                    points=False,
                    marker=dict(
                        color='rgb(90,90,90)',
                        size=4,
                        opacity=0.5,
                    ),
                    width=0.8,
                )
            )

        # ── Trajectoires Min / Déterministe / Max (optionnel) ─────────────────────
        if show_min_max_deterministic and line_data is not None and not line_data.empty:
            line_data.dropna(how='any', inplace=True)
            line_data[year_col] = line_data[year_col].astype(str).str.replace('YEAR_', '', regex=False)
            if years_to_exclude:
                line_data = line_data.loc[~line_data[year_col].isin(years_to_exclude)].copy()
            if line_data.empty:
                line_data = None

        if show_min_max_deterministic and line_data is not None and not line_data.empty:
            line_data['TotalCost_bEUR'] = line_data['TotalCost'] / 1000.0

            sample_total = line_data.groupby(sample_col)['TotalCost_bEUR'].sum()
            if not sample_total.empty:
                min_sample, max_sample, _ = self._get_min_max_samples_from_cost(ampl_uq_collector)
                if min_sample is None or max_sample is None:
                    min_sample = sample_total.idxmin()
                    max_sample = sample_total.idxmax()

                print("Scenarios used for annual Total Cost trajectories:")
                print(f"- Min (UQ): sample={min_sample}")
                print(f"- Max (UQ): sample={max_sample}")

                det_data, det_pkl_path = self._load_deterministic_result('TotalCost')
                if det_data is not None:
                    print(f"- Deterministic: loaded from {det_pkl_path}")
                else:
                    print(f"- Deterministic: unavailable at {det_pkl_path}")

                scenario_specs = [
                    ('Min', min_sample, 'rgb(0,140,0)', None),
                    ('Deterministic', None, 'rgb(100,100,100)', det_data),
                    ('Max', max_sample, 'rgb(200,0,0)', None)
                ]

                for label, sample_id, color, det_df in scenario_specs:
                    if label == 'Deterministic' and det_df is not None:
                        scenario_df = det_df.reset_index()
                        scenario_df.columns = [year_col, 'TotalCost']
                        scenario_df[year_col] = scenario_df[year_col].astype(str).str.replace('YEAR_', '', regex=False)
                        if years_to_exclude:
                            scenario_df = scenario_df.loc[~scenario_df[year_col].isin(years_to_exclude)].copy()
                        if scenario_df.empty:
                            continue
                        scenario_df['TotalCost_bEUR'] = scenario_df['TotalCost'] / 1000.0
                        scenario_df['__order'] = scenario_df[year_col].map(_year_sort_key)
                        scenario_df.sort_values(by='__order', inplace=True)
                    else:
                        scenario_df, _ = self._match_rows_by_sample_strict(line_data, sample_col, sample_id)
                        scenario_df = scenario_df.copy()
                        if scenario_df.empty:
                            continue
                        scenario_df['__order'] = scenario_df[year_col].map(_year_sort_key)
                        scenario_df.sort_values(by='__order', inplace=True)

                    fig.add_trace(
                        go.Scatter(
                            x=scenario_df[year_col],
                            y=scenario_df['TotalCost_bEUR'],
                            #mode='lines+markers',
                            mode='markers',
                            name=label,
                            marker=dict(size=7, color=color, line=dict(color='white', width=1)),
                            legendgroup=label,
                        )
                    )

        if show_det:
            det_path = "/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/Deterministic (mean value)/_Results.pkl"
            try:
                with open(det_path, 'rb') as f:
                    det_results = pkl.load(f)
                if 'TotalCost' in det_results:
                    det_df = det_results['TotalCost'].copy().reset_index()
                    det_df.columns = [year_col, 'TotalCost']
                    det_df[year_col] = det_df[year_col].astype(str).str.replace('YEAR_', '', regex=False)
                    if years_to_exclude:
                        det_df = det_df.loc[~det_df[year_col].isin(years_to_exclude)].copy()
                    det_df['TotalCost_bEUR'] = det_df['TotalCost'] / 1000.0
                    det_df['__order'] = det_df[year_col].map(_year_sort_key)
                    det_df.sort_values(by='__order', inplace=True)
                    fig.add_trace(go.Scatter(
                        x=det_df[year_col],
                        y=det_df['TotalCost_bEUR'],
                        mode='lines+markers',
                        name='DET',
                        line=dict(color='rgba(60,60,60,0.85)', width=1.5),
                        marker=dict(size=5, color='rgba(60,60,60,0.85)'),
                    ))
            except Exception as e:
                print(f"DET scenario unavailable: {e}")

        fig.update_layout(
            title='Annual total cost by scenario',
            violinmode='overlay',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years),
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "TotalCost/")):
            Path(self.outdir + "TotalCost/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "TotalCost/_Raw/")):
            Path(self.outdir + "TotalCost/_Raw/").mkdir(parents=True, exist_ok=True)

        fig.write_html(self.outdir + "TotalCost/_Raw/TotalCost_per_year_scenarios_raw.html")

        y_min = float(results['TotalCost_bEUR'].min())
        y_max = float(results['TotalCost_bEUR'].max())
        yvals = [round(y_min, 2), round(y_max, 2)] if y_min != y_max else [round(y_min, 2)]

        title = "<b>Annual total cost across scenarios</b><br>[b€]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        first_year = ordered_years[0] if ordered_years else None
        fig.update_xaxes(
            tickvals=ordered_years,
            ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
        )
        fig.update_layout(
            showlegend=False,
            legend=dict(
                x=0.99,
                y=0.99,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(90,90,90,0.3)',
                borderwidth=1
            )
        )
        fig.write_image(self.outdir + "TotalCost/TotalCost_per_year_scenarios.pdf", width=1200, height=550)
        plt.close()

        summary = (
            results
            .groupby(year_col, as_index=False)['TotalCost_bEUR']
            .agg(['count', 'mean', 'median', 'min', 'max'])
            .reset_index()
            .rename(columns={year_col: 'Year'})
        )
        summary.to_csv(self.outdir + "TotalCost/TotalCost_per_year_scenarios_summary.csv", index=False)

        return results


    def graph_inv_cost_per_year_scenarios(self, ampl_uq_collector=None, plot=True,
                                          exclude_years=None, exclude_2020=False,
                                          show_min_max_deterministic=False,
                                          show_det=False,
                                          y_max=None):
        """
        Violin plot du coût d'investissement annuel par scénario UQ.

        La phase '2020_2025' est attribuée à l'année 2020 (année de début).

        Parameters
        ----------
        exclude_years : list[str|int] | None
            Années de début à exclure (ex: [2020, '2030']).
        exclude_2020 : bool
            Raccourci pour exclure l'année 2020.
        show_min_max_deterministic : bool
            Si True, superpose les trajectoires Min / Déterministe / Max.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_inv_phase' not in ampl_uq_collector:
            raise ValueError("'C_inv_phase' introuvable dans le collecteur UQ.")

        results = ampl_uq_collector['C_inv_phase'].copy().reset_index()

        phase_col = 'Phases' if 'Phases' in results.columns else results.columns[0]
        sample_col = 'Sample' if 'Sample' in results.columns else None
        if 'C_inv_phase' not in results.columns:
            raise ValueError("Colonne 'C_inv_phase' introuvable dans 'C_inv_phase'.")

        line_data = None
        if sample_col is not None:
            line_data = results[[sample_col, phase_col, 'C_inv_phase']].copy()

        results = results[[phase_col, 'C_inv_phase']].copy()
        results.dropna(how='any', inplace=True)

        # Phase → année de début (ex: '2020_2025' → '2020')
        results['Year'] = results[phase_col].astype(str).str.split('_').str[0]

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).split('_')[0] for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')

        if years_to_exclude:
            results = results.loc[~results['Year'].isin(years_to_exclude)].copy()

        if results.empty:
            raise ValueError("Aucune donnée restante après exclusion des années demandées.")

        results['C_inv_bEUR'] = results['C_inv_phase'] / 1000.0

        def _year_sort_key(value):
            nums = re.findall(r'\d+', str(value))
            return int(nums[0]) if nums else 10**9

        ordered_years = sorted(results['Year'].unique().tolist(), key=_year_sort_key)

        fig = go.Figure()

        for year in ordered_years:
            year_vals = results.loc[results['Year'] == year, 'C_inv_bEUR']
            fig.add_trace(
                go.Box(
                    x=[year] * len(year_vals),
                    y=year_vals,
                    name=year,
                    showlegend=False,
                    fillcolor='rgba(90,90,90,0.25)',
                    line=dict(color='rgb(90,90,90)'),
                    boxpoints=False,
                    width=0.8,
                )
            )

        if show_min_max_deterministic and line_data is not None and not line_data.empty:
            line_data.dropna(how='any', inplace=True)
            line_data['Year'] = line_data[phase_col].astype(str).str.split('_').str[0]
            if years_to_exclude:
                line_data = line_data.loc[~line_data['Year'].isin(years_to_exclude)].copy()

        if show_min_max_deterministic and line_data is not None and not line_data.empty:
            line_data['C_inv_bEUR'] = line_data['C_inv_phase'] / 1000.0

            sample_total = line_data.groupby(sample_col)['C_inv_bEUR'].sum()
            if not sample_total.empty:
                min_sample, max_sample, _ = self._get_min_max_samples_from_cost(ampl_uq_collector)
                if min_sample is None or max_sample is None:
                    min_sample = sample_total.idxmin()
                    max_sample = sample_total.idxmax()

                det_data, det_pkl_path = self._load_deterministic_result('C_inv_phase')
                print("Scenarios used for investment cost trajectories:")
                print(f"- Min (UQ): sample={min_sample}")
                print(f"- Max (UQ): sample={max_sample}")
                if det_data is not None:
                    print(f"- Deterministic: loaded from {det_pkl_path}")
                else:
                    print(f"- Deterministic: unavailable at {det_pkl_path}")

                scenario_specs = [
                    ('Min', min_sample, 'rgb(0,140,0)', None),
                    ('Deterministic', None, 'rgb(100,100,100)', det_data),
                    ('Max', max_sample, 'rgb(200,0,0)', None),
                ]

                for label, sample_id, color, det_df in scenario_specs:
                    if label == 'Deterministic' and det_df is not None:
                        scenario_df = det_df.reset_index()
                        scenario_df.columns = [phase_col, 'C_inv_phase']
                        scenario_df['Year'] = scenario_df[phase_col].astype(str).str.split('_').str[0]
                        if years_to_exclude:
                            scenario_df = scenario_df.loc[~scenario_df['Year'].isin(years_to_exclude)].copy()
                        if scenario_df.empty:
                            continue
                        scenario_df['C_inv_bEUR'] = scenario_df['C_inv_phase'] / 1000.0
                        scenario_df['__order'] = scenario_df['Year'].map(_year_sort_key)
                        scenario_df.sort_values(by='__order', inplace=True)
                    else:
                        scenario_df, _ = self._match_rows_by_sample_strict(line_data, sample_col, sample_id)
                        scenario_df = scenario_df.copy()
                        if scenario_df.empty:
                            continue
                        scenario_df['__order'] = scenario_df['Year'].map(_year_sort_key)
                        scenario_df.sort_values(by='__order', inplace=True)

                    fig.add_trace(
                        go.Scatter(
                            x=scenario_df['Year'],
                            y=scenario_df['C_inv_bEUR'],
                            mode='markers',
                            name=label,
                            marker=dict(size=7, color=color, line=dict(color='white', width=1)),
                            legendgroup=label,
                        )
                    )

        if show_det:
            det_paths = {
                'DET': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/Deterministic (mean value)/_Results.pkl',
            }
            det_colors = {'DET': 'rgba(60,60,60,0.85)'}
            det_dash   = {'DET': 'solid'}
            for label, path in det_paths.items():
                try:
                    with open(path, 'rb') as f:
                        det_res = pkl.load(f)
                    if 'C_inv_phase' not in det_res:
                        print(f"{label}: 'C_inv_phase' introuvable dans {path}")
                        continue
                    det_df = det_res['C_inv_phase'].copy().reset_index()
                    det_phase_col = 'Phases' if 'Phases' in det_df.columns else det_df.columns[0]
                    det_df['Year'] = det_df[det_phase_col].astype(str).str.split('_').str[0]
                    if years_to_exclude:
                        det_df = det_df[~det_df['Year'].isin(years_to_exclude)]
                    det_df['C_inv_bEUR'] = det_df['C_inv_phase'] / 1000.0
                    det_df['__order'] = det_df['Year'].map(_year_sort_key)
                    det_df.sort_values('__order', inplace=True)
                    fig.add_trace(go.Scatter(
                        x=det_df['Year'],
                        y=det_df['C_inv_bEUR'],
                        mode='lines+markers',
                        name=label,
                        line=dict(color=det_colors[label], width=1.5, dash=det_dash[label]),
                        marker=dict(size=5, color=det_colors[label]),
                    ))
                except Exception as e:
                    print(f"{label} unavailable: {e}")

        fig.update_layout(
            title='Annual investment cost by scenario',
            violinmode='overlay',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years),
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "InvCost/")):
            Path(self.outdir + "InvCost/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "InvCost/_Raw/")):
            Path(self.outdir + "InvCost/_Raw/").mkdir(parents=True, exist_ok=True)

        fig.write_html(self.outdir + "InvCost/_Raw/InvCost_per_year_scenarios_raw.html")

        y_min = float(results['C_inv_bEUR'].min())
        _y_max = y_max if y_max is not None else float(results['C_inv_bEUR'].max())
        yvals = [round(y_min, 2), round(_y_max, 2)] if y_min != _y_max else [round(y_min, 2)]

        title = "<b>Annual investment cost across scenarios</b><br>[b€]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        first_year = ordered_years[0] if ordered_years else None
        fig.update_xaxes(
            tickvals=ordered_years,
            ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
        )
        fig.update_layout(
            legend=dict(
                x=0.99, y=0.99,
                xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(90,90,90,0.3)',
                borderwidth=1,
            )
        )
        fig.write_image(self.outdir + "InvCost/InvCost_per_year_scenarios.pdf", width=1200, height=550)
        plt.close()

        summary = (
            results
            .groupby('Year', as_index=False)['C_inv_bEUR']
            .agg(['count', 'mean', 'median', 'min', 'max'])
            .reset_index()
        )
        summary.to_csv(self.outdir + "InvCost/InvCost_per_year_scenarios_summary.csv", index=False)

        return results


    def graph_capex_opex_per_year_scenarios(self, ampl_uq_collector=None, plot=True,
                                             exclude_years=None, exclude_2020=False,
                                             y_max=None):
        """
        Violin plots côte à côte du CAPEX (C_inv_phase) et de l'OPEX total
        (C_op_phase_res sommé sur les ressources) par phase/année, sur tous
        les scénarios UQ.

        La phase '2020_2025' est attribuée à l'année 2020 (année de début).

        Parameters
        ----------
        exclude_years : list[str|int] | None
            Années de début à exclure.
        exclude_2020 : bool
            Raccourci pour exclure l'année 2020.
        y_max : float | None
            Plafond de l'axe Y en b€ (auto si None).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        for key in ('C_inv_phase', 'C_op_phase_res'):
            if key not in ampl_uq_collector:
                raise ValueError(f"'{key}' introuvable dans le collecteur UQ.")

        def _year_sort_key(value):
            nums = re.findall(r'\d+', str(value))
            return int(nums[0]) if nums else 10**9

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).split('_')[0] for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')

        # ── CAPEX : C_inv_phase (index = Phases) ─────────────────────────────────
        capex = ampl_uq_collector['C_inv_phase'].copy().reset_index()
        phase_col = 'Phases' if 'Phases' in capex.columns else capex.columns[0]
        capex['Year'] = capex[phase_col].astype(str).str.split('_').str[0]
        capex.dropna(subset=['C_inv_phase'], inplace=True)
        capex['C_inv_bEUR'] = pd.to_numeric(capex['C_inv_phase'], errors='coerce') / 1000.0
        capex.dropna(subset=['C_inv_bEUR'], inplace=True)
        if years_to_exclude:
            capex = capex[~capex['Year'].isin(years_to_exclude)]

        # ── OPEX : C_op_phase_res (MultiIndex Phases × Resources) ────────────────
        opex = ampl_uq_collector['C_op_phase_res'].copy().reset_index()
        op_phase_col = 'Phases' if 'Phases' in opex.columns else opex.columns[0]
        opex['Year'] = opex[op_phase_col].astype(str).str.split('_').str[0]
        opex['C_op_phase_res'] = pd.to_numeric(opex['C_op_phase_res'], errors='coerce').fillna(0)
        opex_agg = opex.groupby(['Sample', 'Year'], as_index=False)['C_op_phase_res'].sum()
        opex_agg['C_op_bEUR'] = opex_agg['C_op_phase_res'] / 1000.0
        if years_to_exclude:
            opex_agg = opex_agg[~opex_agg['Year'].isin(years_to_exclude)]

        if capex.empty or opex_agg.empty:
            raise ValueError("Aucune donnée restante après exclusion des années demandées.")

        ordered_years = sorted(
            set(capex['Year'].unique()) | set(opex_agg['Year'].unique()),
            key=_year_sort_key
        )

        # ── Figure : CAPEX / OPEX côte à côte via px.box ────────────────────────
        df_cap = capex[['Sample', 'Year', 'C_inv_bEUR']].copy()
        df_cap.rename(columns={'C_inv_bEUR': 'Value_bEUR'}, inplace=True)
        df_cap['Cost_type'] = 'CAPEX'

        df_op = opex_agg[['Sample', 'Year', 'C_op_bEUR']].copy()
        df_op.rename(columns={'C_op_bEUR': 'Value_bEUR'}, inplace=True)
        df_op['Cost_type'] = 'OPEX'

        df_long = pd.concat([df_cap, df_op], ignore_index=True)
        df_long = df_long[df_long['Year'].isin(ordered_years)]

        fig = px.box(
            df_long, x='Year', y='Value_bEUR', color='Cost_type',
            color_discrete_map={'CAPEX': 'rgb(60,100,180)', 'OPEX': 'rgb(200,80,40)'},
            points='outliers', notched=False,
            category_orders={'Year': ordered_years},
        )
        fig.update_layout(
            title='CAPEX vs OPEX by year across scenarios',
            boxmode='group',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "CapexOpex/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(out_dir + "_Raw/CapexOpex_per_year_scenarios_raw.html")

        all_vals = pd.concat([capex['C_inv_bEUR'], opex_agg['C_op_bEUR']])
        y_min = float(all_vals.min())
        _y_max = y_max if y_max is not None else float(all_vals.max())
        yvals = [round(y_min, 2), round(_y_max, 2)] if y_min != _y_max else [round(y_min, 2)]

        title = "<b>CAPEX vs OPEX across scenarios</b><br>[b€]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        first_year = ordered_years[0] if ordered_years else None
        fig.update_xaxes(
            tickvals=ordered_years,
            ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
        )
        fig.update_layout(
            showlegend=False,
            legend=dict(
                x=0.99, y=0.99,
                xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(90,90,90,0.3)',
                borderwidth=1,
            )
        )
        fig.write_image(out_dir + "CapexOpex_per_year_scenarios.pdf", width=1200, height=550)
        plt.close()

        return capex, opex_agg


    def graph_total_capex_opex_scenarios(self, ampl_uq_collector=None, plot=True,
                                          y_max=None, show_det=True, show_nct_only=False):
        """
        Deux box plots côte à côte montrant la distribution du CAPEX total
        et de l'OPEX total à travers les scénarios UQ (depuis C_tot_capex / C_tot_opex).

        Parameters
        ----------
        y_max : float | None
            Plafond de l'axe Y en b€ (auto si None).
        show_det : bool
            Si True, superpose les valeurs DET et NCT comme croix.
        show_nct_only : bool
            Si True, ne superpose que la référence NCT (sans DET).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        for key in ('C_tot_capex', 'C_tot_opex'):
            if key not in ampl_uq_collector:
                raise ValueError(f"'{key}' introuvable dans le collecteur UQ.")

        # ── CAPEX total par sample ────────────────────────────────────────────────
        capex = ampl_uq_collector['C_tot_capex'].copy().reset_index()
        capex_val_col = 'C_tot_capex' if 'C_tot_capex' in capex.columns else capex.columns[-1]
        capex_total = capex[['Sample', capex_val_col]].copy()
        capex_total['C_inv_bEUR'] = pd.to_numeric(capex_total[capex_val_col], errors='coerce').fillna(0) / 1000.0

        # ── OPEX total par sample ─────────────────────────────────────────────────
        opex = ampl_uq_collector['C_tot_opex'].copy().reset_index()
        opex_val_col = 'C_tot_opex' if 'C_tot_opex' in opex.columns else opex.columns[-1]
        opex_total = opex[['Sample', opex_val_col]].copy()
        opex_total['C_op_bEUR'] = pd.to_numeric(opex_total[opex_val_col], errors='coerce').fillna(0) / 1000.0

        if capex_total.empty or opex_total.empty:
            raise ValueError("Aucune donnée trouvée dans C_tot_capex / C_tot_opex.")

        # ── Figure ────────────────────────────────────────────────────────────────
        box_style = dict(
            boxpoints='outliers',
            width=0.4,
        )

        fig = go.Figure()
        fig.add_trace(go.Box(
            x=['Investment costs'] * len(capex_total),
            y=capex_total['C_inv_bEUR'],
            name='Investment costs',
            fillcolor='rgba(60,100,180,0.25)',
            line_color='rgb(60,100,180)',
            marker=dict(color='rgb(60,100,180)', size=4),
            **box_style,
        ))
        fig.add_trace(go.Box(
            x=['Operational costs'] * len(opex_total),
            y=opex_total['C_op_bEUR'],
            name='Operational costs',
            fillcolor='rgba(200,80,40,0.25)',
            line_color='rgb(200,80,40)',
            marker=dict(color='rgb(200,80,40)', size=4),
            **box_style,
        ))

        # ── DET / NCT markers ─────────────────────────────────────────────────────
        det_tick_vals = []   # y-values to add as extra ticks
        det_tick_text = []   # corresponding tick labels
        det_paths = {}
        if show_nct_only:
            det_paths = {
                'NCT': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/No commissioning/_Results.pkl',
            }
        elif show_det:
            det_paths = {
                'DET': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/Deterministic (mean value)/_Results.pkl',
                'NCT': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/No commissioning/_Results.pkl',
            }
        det_colors = {'DET': 'rgb(60,60,60)', 'NCT': 'rgb(180,60,60)'}
        if det_paths:
            for label, path in det_paths.items():
                try:
                    with open(path, 'rb') as f:
                        det_res = pkl.load(f)

                    def _scalar_beur(data):
                        if isinstance(data, pd.DataFrame):
                            s = data.stack()
                        elif isinstance(data, pd.Series):
                            s = data
                        else:
                            s = pd.Series([data])
                        s = pd.to_numeric(s, errors='coerce').dropna()
                        return float(s.iloc[0]) / 1000.0 if len(s) > 0 else None

                    det_capex_val = _scalar_beur(det_res['C_tot_capex']) if 'C_tot_capex' in det_res else None
                    det_opex_val  = _scalar_beur(det_res['C_tot_opex'])  if 'C_tot_opex'  in det_res else None

                    color = det_colors[label]
                    if det_capex_val is not None:
                        fig.add_trace(go.Scatter(
                            x=['Investment costs'], y=[det_capex_val],
                            mode='markers',
                            marker=dict(symbol='x', size=14, color=color, line=dict(width=2, color=color)),
                            name=label, showlegend=True, legendgroup=label,
                        ))
                        if label == 'DET':
                            det_tick_vals.append(round(det_capex_val, 2))
                            det_tick_text.append(f"{round(det_capex_val, 2):.2f}")
                    if det_opex_val is not None:
                        fig.add_trace(go.Scatter(
                            x=['Operational costs'], y=[det_opex_val],
                            mode='markers',
                            marker=dict(symbol='x', size=14, color=color, line=dict(width=2, color=color)),
                            name=label, showlegend=False, legendgroup=label,
                        ))
                        det_tick_vals.append(round(det_opex_val, 2))
                        det_tick_text.append(f"{round(det_opex_val, 2):.2f}")
                except Exception:
                    pass

        fig.update_layout(
            title='Total CAPEX vs OPEX across scenarios',
            boxmode='group',
            showlegend=bool(det_paths),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "CapexOpex/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(out_dir + "_Raw/Total_CapexOpex_scenarios_raw.html")

        all_vals = pd.concat([capex_total['C_inv_bEUR'], opex_total['C_op_bEUR']])
        y_min = float(all_vals.min())
        _y_max = y_max if y_max is not None else float(all_vals.max())
        yvals = [round(y_min, 2), round(_y_max, 2)] if y_min != _y_max else [round(y_min, 2)]

        title = "<b>Total CAPEX vs OPEX across scenarios</b><br>[b€]"
        self.custom_fig(fig, title, yvals, xvals=['Investment costs', 'Operational costs'], type_graph='bar')

        capex_min    = round(float(capex_total['C_inv_bEUR'].min()),    2)
        capex_median = round(float(capex_total['C_inv_bEUR'].median()), 2)
        capex_max    = round(float(capex_total['C_inv_bEUR'].max()),    2)
        opex_min     = round(float(opex_total['C_op_bEUR'].min()),      2)
        opex_median  = round(float(opex_total['C_op_bEUR'].median()),   2)
        opex_max     = round(float(opex_total['C_op_bEUR'].max()),      2)

        stat_vals = sorted(set([capex_min, capex_median, capex_max,
                                opex_min,  opex_median,  opex_max]))
        stat_text = [f"{v:.2f}" for v in stat_vals]

        fig.update_yaxes(tickvals=stat_vals, ticktext=stat_text)

        fig.write_image(out_dir + "Total_CapexOpex_scenarios.pdf", width=600, height=550)
        plt.close()

        summary = pd.DataFrame({
            'Metric': ['CAPEX', 'OPEX'],
            'mean':   [capex_total['C_inv_bEUR'].mean(),  opex_total['C_op_bEUR'].mean()],
            'median': [capex_total['C_inv_bEUR'].median(), opex_total['C_op_bEUR'].median()],
            'min':    [capex_total['C_inv_bEUR'].min(),   opex_total['C_op_bEUR'].min()],
            'max':    [capex_total['C_inv_bEUR'].max(),   opex_total['C_op_bEUR'].max()],
        })
        summary.to_csv(out_dir + "Total_CapexOpex_scenarios_summary.csv", index=False)

        return capex_total, opex_total


    def graph_elec_production_per_year_scenarios(self, ampl_uq_collector=None, plot=True,
                                                  exclude_years=None, exclude_2020=False):
        """
        Violin plot de la production annuelle d'électricité (somme des valeurs positives
        du layer ELECTRICITY) par année, sur tous les samples UQ.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        df = ampl_uq_collector['Year_balance'][['ELECTRICITY', 'Sample']].copy().reset_index()

        year_col    = 'Years'
        sample_col  = 'Sample'

        df[year_col] = df[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).replace('YEAR_', '') for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')
        if years_to_exclude:
            df = df.loc[~df[year_col].isin(years_to_exclude)].copy()

        # Somme des valeurs positives uniquement (production)
        df_prod = df[df['ELECTRICITY'] > 0].copy()
        df_prod = df_prod.groupby([year_col, sample_col], as_index=False)['ELECTRICITY'].sum()
        df_prod['Elec_TWh'] = df_prod['ELECTRICITY'] / 1000.0

        if df_prod.empty:
            raise ValueError("Aucune valeur positive d'électricité trouvée.")

        def _year_sort_key(v):
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else 10**9

        ordered_years = sorted(df_prod[year_col].unique().tolist(), key=_year_sort_key)

        fig = go.Figure()
        for year in ordered_years:
            vals = df_prod.loc[df_prod[year_col] == year, 'Elec_TWh']
            fig.add_trace(go.Violin(
                x=[year] * len(vals),
                y=vals,
                name=year,
                showlegend=False,
                box_visible=False,
                meanline_visible=False,
                fillcolor='rgba(90,90,90,0.25)',
                line_color='rgb(90,90,90)',
                points=False,
                width=0.8,
            ))

        # ── END_USES demand (médiane par année) ───────────────────────────────
        df_raw = ampl_uq_collector['Year_balance'][['ELECTRICITY', 'Sample']].copy().reset_index()
        df_raw[year_col] = df_raw[year_col].astype(str).str.replace('YEAR_', '', regex=False)
        if years_to_exclude:
            df_raw = df_raw[~df_raw[year_col].isin(years_to_exclude)]
        eud_mask = df_raw['Elements'].astype(str).str.contains('END_USES', case=False)
        df_eud = df_raw[eud_mask].copy()
        df_eud['Elec_TWh'] = df_eud['ELECTRICITY'].clip(upper=0).abs() / 1000.0
        df_eud = df_eud[df_eud['Elec_TWh'] > 0].groupby(
            [year_col, sample_col], as_index=False
        )['Elec_TWh'].sum()
        df_eud_med = df_eud.groupby(year_col, as_index=False)['Elec_TWh'].median()
        df_eud_med['__order'] = df_eud_med[year_col].map(_year_sort_key)
        df_eud_med = df_eud_med[df_eud_med[year_col].isin(ordered_years)].sort_values('__order')
        if not df_eud_med.empty:
            fig.add_trace(go.Scatter(
                x=df_eud_med[year_col], y=df_eud_med['Elec_TWh'],
                mode='lines+markers',
                name='END_USES',
                line=dict(color='steelblue', width=2, dash='dash'),
                marker=dict(size=4, color='steelblue'),
                showlegend=False,
            ))

        # ── DET / NCT trajectoires légères ────────────────────────────────────────
        det_paths = {
            'DET': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/Deterministic (mean value)/_Results.pkl',
            'NCT': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/No commissioning/_Results.pkl',
        }
        det_colors = {'DET': 'rgba(60,60,60,0.5)', 'NCT': 'rgba(180,60,60,0.5)'}

        for label, path in det_paths.items():
            try:
                with open(path, 'rb') as f:
                    det_res = pkl.load(f)
                if 'Year_balance' not in det_res:
                    continue
                det_df = det_res['Year_balance'][['ELECTRICITY']].copy().reset_index()
                det_df['Years'] = det_df['Years'].astype(str).str.replace('YEAR_', '', regex=False)
                if years_to_exclude:
                    det_df = det_df[~det_df['Years'].isin(years_to_exclude)]
                det_df = det_df[det_df['ELECTRICITY'] > 0].groupby('Years', as_index=False)['ELECTRICITY'].sum()
                det_df['Elec_TWh'] = det_df['ELECTRICITY'] / 1000.0
                det_df['__order'] = det_df['Years'].map(_year_sort_key)
                det_df.sort_values('__order', inplace=True)
                fig.add_trace(go.Scatter(
                    x=det_df['Years'], y=det_df['Elec_TWh'],
                    mode='lines+markers',
                    line=dict(color=det_colors[label], width=1.5),
                    marker=dict(size=4, color=det_colors[label]),
                    showlegend=False,
                ))
            except Exception:
                pass

        fig.update_layout(
            title='Annual electricity production by scenario',
            violinmode='overlay',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "Electricity/"
        if not os.path.exists(Path(outdir)):
            Path(outdir).mkdir(parents=True, exist_ok=True)

        fig.write_html(outdir + "Elec_production_per_year_scenarios_raw.html")

        y_min = 76.6

        #y_max = float(df_prod['Elec_TWh'].max())
        y_max = float(250.6)
        yvals = [round(y_min, 1), round(y_max, 1)] if y_min != y_max else [round(y_min, 1)]

        df_2033 = df_prod.loc[df_prod[year_col] == '2033', 'Elec_TWh']
        if not df_2033.empty:
            yvals = sorted(set(yvals) | {round(float(df_2033.min()), 1), round(float(df_2033.max()), 1)})

        df_2041 = df_prod.loc[df_prod[year_col] == '2041', 'Elec_TWh']
        if not df_2041.empty:
            yvals = sorted(set(yvals) | {round(float(df_2041.min()), 1), round(float(df_2041.max()), 1)})

        title = "<b>Annual electricity production across scenarios</b><br>[TWh]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        first_year = ordered_years[0] if ordered_years else None
        fig.update_xaxes(
            tickvals=ordered_years,
            ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
        )
        fig.write_image(outdir + "Elec_production_per_year_scenarios.pdf", width=1200, height=550)
        plt.close()

        return df_prod


    def graph_co2_captured_per_year_scenarios(self, ampl_uq_collector=None, plot=True,
                                               exclude_years=None, exclude_2020=False):
        """
        Violin plot de la variation annuelle de CO2_CAPTURED (Year_balance)
        par année, sur tous les samples UQ.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        if 'CO2_CAPTURED' not in yb.columns:
            raise ValueError("Colonne 'CO2_CAPTURED' introuvable dans Year_balance.")

        year_col   = 'Years'
        sample_col = 'Sample'

        yb[year_col] = yb[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).replace('YEAR_', '') for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')
        if years_to_exclude:
            yb = yb.loc[~yb[year_col].isin(years_to_exclude)].copy()

        df = yb[yb['CO2_CAPTURED'] > 0].copy()
        df = df.groupby([year_col, sample_col], as_index=False)['CO2_CAPTURED'].sum()
        df['CO2_MtCO2'] = df['CO2_CAPTURED'] / 1000.0

        if df.empty:
            raise ValueError("Aucune valeur positive de CO2_CAPTURED trouvée.")

        def _year_sort_key(v):
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else 10**9

        ordered_years = sorted(
            [y for y in df[year_col].unique() if _year_sort_key(y) >= 2040],
            key=_year_sort_key
        )
        df = df[df[year_col].isin(ordered_years)].copy()

        fig = go.Figure()
        for year in ordered_years:
            vals = df.loc[df[year_col] == year, 'CO2_MtCO2']
            fig.add_trace(go.Box(
                x=[year] * len(vals),
                y=vals,
                name=year,
                showlegend=False,
                fillcolor='rgba(90,90,90,0.25)',
                line_color='rgb(90,90,90)',
                marker=dict(color='rgb(90,90,90)', size=3),
                boxpoints='outliers',
                width=0.6,
            ))

        fig.update_layout(
            title='Annual CO2 captured by scenario',
            boxmode='overlay',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "CO2/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)

        fig.write_html(outdir + "_Raw/CO2_captured_per_year_scenarios_raw.html")

        y_max = float(df['CO2_MtCO2'].max())
        yvals = [0, round(y_max, 1)]

        title = "<b>Annual CO2 captured across scenarios</b><br>[MtCO2]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        fig.update_yaxes(rangemode='nonnegative')
        first_year = ordered_years[0] if ordered_years else None
        fig.update_xaxes(
            tickvals=ordered_years,
            ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
        )
        fig.write_image(outdir + "CO2_captured_per_year_scenarios.pdf", width=1200, height=550)
        plt.close()

        return df

    def graph_co2_captured_total_scenarios(self, ampl_uq_collector=None, plot=True,
                                            year_start=None, year_end=None):
        """
        Box plot de la somme totale de CO2 capturé par scénario (un seul box,
        une valeur par sample = somme sur toutes les années).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()
        if 'CO2_CAPTURED' not in yb.columns:
            raise ValueError("Colonne 'CO2_CAPTURED' introuvable dans Year_balance.")

        def _year_sort_key(v):
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else 10**9

        yb['Years'] = yb['Years'].astype(str).str.replace('YEAR_', '', regex=False)
        yb['CO2_CAPTURED'] = pd.to_numeric(yb['CO2_CAPTURED'], errors='coerce').fillna(0)

        if year_start is not None:
            yb = yb[yb['Years'].apply(_year_sort_key) >= int(year_start)]
        if year_end is not None:
            yb = yb[yb['Years'].apply(_year_sort_key) <= int(year_end)]

        df = yb[yb['CO2_CAPTURED'] > 0].groupby('Sample', as_index=False)['CO2_CAPTURED'].sum()
        df['CO2_MtCO2'] = df['CO2_CAPTURED'] / 1000.0

        if df.empty:
            raise ValueError("Aucune valeur positive de CO2_CAPTURED trouvée.")

        if not plot:
            return df

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df['CO2_MtCO2'],
            name='Total CO2 captured',
            fillcolor='rgba(90,90,90,0.25)',
            line_color='rgb(90,90,90)',
            marker=dict(color='rgb(90,90,90)', size=4),
            boxpoints='all',
            jitter=0.3,
            pointpos=0,
        ))

        fig.update_layout(
            title='Total CO2 captured across scenarios',
            template='simple_white',
            showlegend=False,
        )
        pio.show(fig)

        outdir = self.outdir + "CO2/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/CO2_captured_total_scenarios_raw.html")

        y_max = float(df['CO2_MtCO2'].max())
        yvals = [0, round(y_max, 1)]
        title_str = "<b>Total CO2 captured across scenarios</b><br>[MtCO2]"
        if year_start or year_end:
            title_str = f"<b>Total CO2 captured ({year_start or ''}–{year_end or ''}) across scenarios</b><br>[MtCO2]"
        self.custom_fig(fig, title_str, yvals, type_graph='bar')
        fig.update_layout(showlegend=False)
        fig.write_image(outdir + "CO2_captured_total_scenarios.pdf", width=500, height=550)
        plt.close()

        return df


    def graph_heat_production_per_year_scenarios(self, ampl_uq_collector=None, plot=True,
                                                  exclude_years=None, exclude_2020=False):
        """
        Violin plot de la production annuelle de chaleur (somme des valeurs positives)
        par année, sur tous les samples UQ.

        Produit deux figures séparées :
          - Low Heat (LT) : HEAT_LOW_T_DHN + HEAT_LOW_T_DECEN
          - High Heat (HT) : HEAT_HIGH_T
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        year_col   = 'Years'
        sample_col = 'Sample'

        yb[year_col] = yb[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).replace('YEAR_', '') for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')
        if years_to_exclude:
            yb = yb.loc[~yb[year_col].isin(years_to_exclude)].copy()

        def _year_sort_key(v):
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else 10**9

        lt_cols = [c for c in yb.columns if 'HEAT_LOW_T_DHN'   in str(c)] + \
                  [c for c in yb.columns if 'HEAT_LOW_T_DECEN' in str(c)]
        ht_cols = [c for c in yb.columns if 'HEAT_HIGH_T'      in str(c)]

        outdir = self.outdir + "Heat/"
        Path(outdir).mkdir(parents=True, exist_ok=True)

        results = {}
        for tag, label, heat_cols in [
            ('LT', 'Low Heat',  lt_cols),
            ('HT', 'High Heat', ht_cols),
        ]:
            if not heat_cols:
                continue

            df = yb[[year_col, sample_col] + heat_cols].copy()
            df['Heat_TWh'] = df[heat_cols].clip(lower=0).sum(axis=1) / 1000.0
            df_prod = df[df['Heat_TWh'] > 0].groupby(
                [year_col, sample_col], as_index=False
            )['Heat_TWh'].sum()

            if df_prod.empty:
                continue

            ordered_years = sorted(df_prod[year_col].unique().tolist(), key=_year_sort_key)

            fig = go.Figure()
            for year in ordered_years:
                vals = df_prod.loc[df_prod[year_col] == year, 'Heat_TWh']
                fig.add_trace(go.Violin(
                    x=[year] * len(vals),
                    y=vals,
                    name=year,
                    showlegend=False,
                    box_visible=False,
                    meanline_visible=False,
                    fillcolor='rgba(90,90,90,0.25)',
                    line_color='rgb(90,90,90)',
                    points=False,
                    width=0.8,
                ))

            # ── END_USES demand (médiane par année) ───────────────────────────
            eud_mask = yb['Elements'].astype(str).str.contains('END_USES', case=False)
            df_eud = yb[eud_mask][[year_col, sample_col] + heat_cols].copy()
            df_eud['Heat_TWh'] = df_eud[heat_cols].clip(upper=0).sum(axis=1).abs() / 1000.0
            df_eud = df_eud[df_eud['Heat_TWh'] > 0].groupby(
                [year_col, sample_col], as_index=False
            )['Heat_TWh'].sum()
            df_eud_med = df_eud.groupby(year_col, as_index=False)['Heat_TWh'].median()
            df_eud_med['__order'] = df_eud_med[year_col].map(_year_sort_key)
            df_eud_med = df_eud_med[df_eud_med[year_col].isin(ordered_years)].sort_values('__order')
            if not df_eud_med.empty:
                fig.add_trace(go.Scatter(
                    x=df_eud_med[year_col], y=df_eud_med['Heat_TWh'],
                    mode='lines+markers',
                    name='END_USES',
                    line=dict(color='steelblue', width=2, dash='dash'),
                    marker=dict(size=4, color='steelblue'),
                    showlegend=False,
                ))

            # ── DET / NCT trajectoires ────────────────────────────────────────
            det_paths = {
                'DET': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/Deterministic (mean value)/_Results.pkl',
                'NCT': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/No commissioning/_Results.pkl',
            }
            det_colors = {'DET': 'rgba(60,60,60,0.5)', 'NCT': 'rgba(180,60,60,0.5)'}

            for det_label, path in det_paths.items():
                try:
                    with open(path, 'rb') as f:
                        det_res = pkl.load(f)
                    if 'Year_balance' not in det_res:
                        continue
                    det_yb = det_res['Year_balance'].copy().reset_index()
                    det_yb['Years'] = det_yb['Years'].astype(str).str.replace('YEAR_', '', regex=False)
                    if years_to_exclude:
                        det_yb = det_yb[~det_yb['Years'].isin(years_to_exclude)]
                    det_heat_cols = [c for c in det_yb.columns if any(k in str(c) for k in heat_cols)]
                    if not det_heat_cols:
                        continue
                    det_yb['Heat_TWh'] = det_yb[det_heat_cols].clip(lower=0).sum(axis=1) / 1000.0
                    det_df = det_yb[det_yb['Heat_TWh'] > 0].groupby('Years', as_index=False)['Heat_TWh'].sum()
                    det_df['__order'] = det_df['Years'].map(_year_sort_key)
                    det_df.sort_values('__order', inplace=True)
                    fig.add_trace(go.Scatter(
                        x=det_df['Years'], y=det_df['Heat_TWh'],
                        mode='lines+markers',
                        name=det_label,
                        line=dict(color=det_colors[det_label], width=1.5),
                        marker=dict(size=4, color=det_colors[det_label]),
                        showlegend=False,
                    ))
                except Exception:
                    pass

            fig.update_layout(
                title=f'Annual {label} production by scenario',
                violinmode='overlay',
                xaxis=dict(categoryorder='array', categoryarray=ordered_years),
            )

            if plot:
                pio.show(fig)

            fig.write_html(outdir + f"Heat_{tag}_production_per_year_scenarios_raw.html")

            #y_min = float(df_prod['Heat_TWh'].min())
            y_min = 56.71
            y_max = float(df_prod['Heat_TWh'].max())
            yvals = [round(y_min, 1), round(y_max, 1)] if y_min != y_max else [round(y_min, 1)]

            title = f"<b>Annual {label} production across scenarios</b><br>[TWh]"
            self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
            first_year = ordered_years[0] if ordered_years else None
            fig.update_xaxes(
                tickvals=ordered_years,
                ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
            )
            fig.write_image(outdir + f"Heat_{tag}_production_per_year_scenarios.pdf", width=1200, height=550)
            plt.close()

            results[tag] = df_prod

        return results


    def graph_high_heat_consumption_by_tech(self, ampl_uq_collector=None, plot=True,
                                             exclude_years=None, exclude_2020=False,
                                             min_share=0.01,
                                             exclude_eud_pattern=r'HEAT_HIGH_T|^EUD|DEMAND'):
        """
        Strip plot de la consommation de chaleur haute température (HEAT_HIGH_T)
        par catégorie consommatrice et par année, sur tous les samples UQ.

        Seuls les éléments qui consomment de la chaleur (valeurs négatives dans
        Year_balance) sont conservés. Le nœud de demande end-use est exclu car
        constant.

        Parameters
        ----------
        exclude_years : list | None
            Années à exclure.
        exclude_2020 : bool
            Si True, exclure 2020.
        min_share : float
            Part minimale de la consommation totale pour garder une catégorie.
            Défaut : 1 %.
        exclude_eud_pattern : str
            Regex pour exclure les nœuds de demande end-use.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        year_col   = 'Years'
        elem_col   = 'Elements' if 'Elements' in yb.columns else yb.columns[1]
        sample_col = 'Sample'

        yb[year_col] = yb[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        years_to_exclude = set()
        if exclude_years is not None:
            years_to_exclude = {str(y).replace('YEAR_', '') for y in exclude_years}
        if exclude_2020:
            years_to_exclude.add('2020')
        if years_to_exclude:
            yb = yb.loc[~yb[year_col].isin(years_to_exclude)].copy()

        ht_cols = [c for c in yb.columns if 'HEAT_HIGH_T' in str(c)]
        if not ht_cols:
            raise ValueError("Aucune colonne HEAT_HIGH_T trouvée dans Year_balance.")

        # consommation = valeurs négatives → on prend la valeur absolue
        yb['Heat_HT'] = yb[ht_cols].clip(upper=0).sum(axis=1).abs()

        # exclure les nœuds de demande end-use
        eud_mask = yb[elem_col].astype(str).str.contains(
            exclude_eud_pattern, flags=re.IGNORECASE, regex=True
        )
        yb = yb[~eud_mask].copy()

        # garder uniquement les catégories avec consommation significative
        consumers_total = yb[yb['Heat_HT'] > 0].groupby(elem_col)['Heat_HT'].sum()
        threshold = consumers_total.sum() * float(min_share)
        consumers = consumers_total[consumers_total >= threshold].index.tolist()
        if not consumers:
            raise ValueError(
                "Aucune catégorie consommatrice de High Heat trouvée après filtrage. "
                "Essayez de réduire min_share ou d'ajuster exclude_eud_pattern."
            )

        df = yb[yb[elem_col].isin(consumers) & (yb['Heat_HT'] > 0)].copy()
        df['Heat_TWh'] = df['Heat_HT'] / 1000.0

        df_cons = df.groupby([year_col, elem_col, sample_col], as_index=False)['Heat_TWh'].sum()

        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        df_cons['Tech_label'] = df_cons[elem_col].apply(
            lambda t: meaning.get(str(t), self._fmt_tech_label(str(t)))
        )

        def _year_sort_key(v):
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else 10**9

        ordered_years = sorted(df_cons[year_col].unique().tolist(), key=_year_sort_key)

        fig = px.strip(
            df_cons,
            x=year_col,
            y='Heat_TWh',
            color='Tech_label',
            color_discrete_map=self.color_dict_full,
            stripmode='overlay',
            hover_data={sample_col: True},
        )
        fig.update_traces(jitter=0.35, pointpos=0,
                          marker=dict(size=6, opacity=0.65))
        fig.update_xaxes(categoryorder='array', categoryarray=ordered_years)

        if plot:
            pio.show(fig)

        outdir = self.outdir + "Heat/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)

        fig.write_html(outdir + "_Raw/Heat_HT_consumption_by_tech_per_year_raw.html")

        y_max = float(df_cons['Heat_TWh'].max())
        yvals = [0, round(y_max, 1)]
        title = "<b>High Heat consumption by category across scenarios</b><br>[TWh]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        first_year = ordered_years[0] if ordered_years else None
        fig.update_xaxes(
            tickvals=ordered_years,
            ticktext=[y if (y == first_year or _year_sort_key(y) % 5 == 0) else '' for y in ordered_years]
        )
        fig.write_image(outdir + "Heat_HT_consumption_by_tech_per_year.pdf", width=1200, height=550)
        plt.close()

        return df_cons


    def graph_elec_correlation_sectors(self, ampl_uq_collector=None, year='all',
                                        corr_method='pearson', plot=True,
                                        min_nonzero_share=0.05, min_std=1e-6):
        """
        Corrélation entre la production d'électricité et la production de chaque autre secteur.

        Pour chaque sample, on calcule la somme des valeurs positives de chaque layer
        (Year_balance), puis on corrèle la colonne ELECTRICITY avec toutes les autres.
        Résultat : bar chart horizontal trié par coefficient de corrélation.

        Parameters
        ----------
        year : str|int
            Année analysée ou 'all' pour agréger toutes les années.
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        layers = ['AMMONIA', 'ELECTRICITY', 'GAS', 'H2', 'WOOD', 'WET_BIOMASS',
                  'HEAT_HIGH_T', 'HEAT_LOW_T_DECEN', 'HEAT_LOW_T_DHN',
                  'HVC', 'METHANOL', 'MOB_FREIGHT_BOAT', 'MOB_FREIGHT_RAIL',
                  'MOB_FREIGHT_ROAD', 'MOB_PRIVATE', 'MOB_PUBLIC']

        yb = ampl_uq_collector['Year_balance'].copy()
        available = [l for l in layers if l in yb.columns]
        df = yb[available + ['Sample']].copy().reset_index()

        df['Years'] = df['Years'].astype(str).str.replace('YEAR_', '', regex=False)

        if str(year).lower() != 'all':
            year_text = str(year)
            df = df[df['Years'].isin({year_text, f'YEAR_{year_text}'})]
            if df.empty:
                raise ValueError(f"Aucune donnée pour l'année {year_text}.")

        # Somme des valeurs positives par (sample, layer)
        records = {}
        for layer in available:
            pos = df[df[layer] > 0].groupby('Sample')[layer].sum()
            records[layer] = pos

        pivot = pd.DataFrame(records).fillna(0)

        if 'ELECTRICITY' not in pivot.columns:
            raise ValueError("ELECTRICITY absent des données.")

        # Filtre : écart-type et part non-nulle suffisants
        nonzero = (pivot.abs() > 1e-9).mean()
        std_dev  = pivot.std(ddof=0)
        keep = [c for c in pivot.columns
                if nonzero[c] >= min_nonzero_share and std_dev[c] >= min_std]
        pivot = pivot[keep]

        corr_series = pivot.corr(method=corr_method)['ELECTRICITY'].drop('ELECTRICITY').sort_values()

        meaning  = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        x_labels = [meaning.get(s, s) for s in corr_series.index]
        y_labels = [meaning.get('ELECTRICITY', 'ELECTRICITY')]

        row_matrix = pd.DataFrame(
            [corr_series.values],
            index=['ELECTRICITY'],
            columns=corr_series.index,
        )

        fig = go.Figure(data=go.Heatmap(
            z=row_matrix.values,
            text=np.round(row_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=x_labels,
            y=y_labels,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            zsmooth='best',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='Sector: %{x}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(text="Electricity production — sector correlation",
                       x=0.5, xanchor='center',
                       font=dict(family='Raleway', size=28)),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=120, r=80, t=90, b=120),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "Electricity/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/Elec_corr_sectors_{year_str}.html")
        fig.write_image(outdir + f"Elec_corr_sectors_{year_str}.pdf", width=1100, height=280)
        plt.close()

        return corr_series


    def graph_energy_mix_correlation_matrix(self, ampl_uq_collector=None,
                                             year_start=2030, year_end=2039,
                                             corr_method='pearson', plot=True,
                                             text_threshold=0.5):
        """
        Deux heatmaps rectangulaires de corrélation croisée :
          1. Axe Y : technologies de production électrique
             Axe X : ressources de production de chaleur basse température (LT)
          2. Axe Y : technologies de production électrique
             Axe X : ressources de production de chaleur haute température (HT)

        Parameters
        ----------
        year_start, year_end : int
        corr_method : str  'pearson', 'spearman' or 'kendall'
        text_threshold : float
            Lignes/colonnes dont le max |corrélation| est inférieur à cette
            valeur sont supprimées. Les valeurs en dessous ne sont pas affichées.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        year_col   = 'Years'    if 'Years'    in yb.columns else yb.columns[0]
        elem_col   = 'Elements' if 'Elements' in yb.columns else yb.columns[1]
        sample_col = 'Sample'   if 'Sample'   in yb.columns else None
        if sample_col is None:
            yb['Sample'] = 0
            sample_col = 'Sample'

        yb[year_col] = yb[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        def _to_int(v):
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else None

        yb['_yr'] = yb[year_col].apply(_to_int)
        yb = yb[(yb['_yr'] >= year_start) & (yb['_yr'] <= year_end)].copy()

        if yb.empty:
            raise ValueError(f"Aucune donnée entre {year_start} et {year_end}.")

        if 'ELECTRICITY' not in yb.columns:
            raise ValueError("Colonne 'ELECTRICITY' introuvable dans Year_balance.")

        heat_high_cols = [c for c in yb.columns if 'HEAT_HIGH_T' in str(c)]

        resource_patterns = [
            ('Elec',           r'ELEC|(?<!TH)HP_|_HP|HEAT_PUMP|BEV|PHEV|HEV|TRAM|METRO|TROLLEY|TRAIN_PUB|RAIL'),
            ('Gas',            r'GAS|\bNG\b|_NG_|_NG$|CNG|LNG|THHP'),
            ('Coal',           r'COAL'),
            ('Wood',           r'WOOD|BIOMASS'),
            ('Waste',          r'WASTE|INCIN'),
            ('Oil',            r'OIL|DIESEL|GASOLINE|PETROL|FUEL_OIL|KEROSENE|JET'),
            ('Nuclear',        r'NUCLEAR'),
            ('Solar',          r'SOLAR'),
            ('Hydrogen', r'AMMONIA|HABER|_H2$'),
            ('H2',             r'\bH2\b|FUEL_CELL|_FC_|\bFC\b|FCEV'),
            ('Synthetic',      r'SYN_|HYDROLYSIS'),
            ('Other',          r'.*'),
        ]

        def _infer_resource(elem_name):
            for res, pat in resource_patterns:
                if re.search(pat, str(elem_name), re.IGNORECASE):
                    return res
            return 'Other'

        def _heat_by_resource(heat_cols, layer_label):
            """One series per resource group that produces heat_cols, per sample."""
            if not heat_cols:
                return {}
            groups = {}
            for elem in yb[elem_col].unique():
                sub = yb[yb[elem_col] == elem][heat_cols]
                if np.nansum(sub.clip(lower=0).values) > 0:
                    res = _infer_resource(elem)
                    groups.setdefault(res, []).append(elem)
            result = {}
            for res, elems in groups.items():
                m = yb[elem_col].isin(elems)
                sub = yb[m][[sample_col] + heat_cols].copy()
                sub[heat_cols] = sub[heat_cols].clip(lower=0)
                result[res] = sub.groupby(sample_col)[heat_cols].sum().sum(axis=1)
            return result

        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}

        # --- séries électricité : une par technologie productrice ---
        elec_records = {}
        _cogen_pat = r'COGEN|CHP'
        elec_total = yb[yb['ELECTRICITY'] > 0].groupby(elem_col)['ELECTRICITY'].sum()
        threshold_elec = elec_total.sum() * 0.01
        for elem in elec_total[elec_total >= threshold_elec].index:
            if re.search(_cogen_pat, str(elem), re.IGNORECASE):
                continue
            sub = yb[(yb[elem_col] == elem) & (yb['ELECTRICITY'] > 0)]
            label = meaning.get(str(elem), self._fmt_tech_label(str(elem)))
            elec_records[label] = sub.groupby(sample_col)['ELECTRICITY'].sum()

        if not elec_records:
            raise ValueError("Aucune technologie électrique trouvée dans Year_balance.")

        elec_pivot = pd.DataFrame(elec_records).fillna(0)
        elec_pivot = elec_pivot.loc[:, elec_pivot.std(ddof=0) > 1e-9]
        elec_cols = list(elec_pivot.columns)

        outdir = self.outdir + "SectorCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)

        heat_dhn_cols   = [c for c in yb.columns if 'HEAT_LOW_T_DHN'   in str(c)]
        heat_decen_cols = [c for c in yb.columns if 'HEAT_LOW_T_DECEN' in str(c)]

        results = {}
        for heat_tag, heat_label, heat_records in [
            ('LT', 'Low Heat',  _heat_by_resource(heat_dhn_cols + heat_decen_cols, 'LT')),
            ('HT', 'High Heat', _heat_by_resource(heat_high_cols, 'HT')),
        ]:
            if not heat_records:
                continue

            heat_pivot = pd.DataFrame(heat_records).fillna(0)
            heat_pivot = heat_pivot.loc[:, heat_pivot.std(ddof=0) > 1e-9]
            heat_cols = list(heat_pivot.columns)

            # corrélation croisée : concat → corr complète → slice [elec, heat]
            combined = pd.concat([elec_pivot, heat_pivot], axis=1).fillna(0)
            full_corr = combined.corr(method=corr_method)
            cross = full_corr.loc[
                [c for c in elec_cols if c in full_corr.index],
                [c for c in heat_cols if c in full_corr.columns],
            ]

            # filtrer les lignes/colonnes sans corrélation significative
            keep_rows = cross.abs().max(axis=1) >= text_threshold
            keep_cols = cross.abs().max(axis=0) >= text_threshold
            cross = cross.loc[keep_rows, keep_cols]

            if cross.empty:
                continue

            y_labels = list(cross.index)
            x_labels = list(cross.columns)

            mask = cross.values.copy()
            mask[np.abs(mask) < text_threshold] = np.nan
            text_mask = np.where(np.isnan(mask), '', np.round(mask, 2).astype(str))

            _title = f"Elec production × {heat_label} correlation ({year_start}–{year_end})"
            _stem  = f"EnergyMix_corr_{heat_tag}_{year_start}_{year_end}"

            fig = go.Figure(data=go.Heatmap(
                z=mask,
                text=text_mask,
                texttemplate='%{text}',
                textfont=dict(color='black', size=11),
                x=x_labels,
                y=y_labels,
                zmin=-1, zmax=1,
                colorscale='RdBu_r',
                colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
                hovertemplate='Heat: %{x}<br>Elec: %{y}<br>Corr: %{z:.3f}<extra></extra>',
            ))

            fig.update_layout(
                title=dict(
                    text=_title,
                    x=0.5, xanchor='center',
                    font=dict(family='Raleway', size=28),
                ),
                template='simple_white',
                font=dict(color='rgb(90,90,90)', size=14),
                margin=dict(l=200, r=80, t=90, b=200),
                width=max(600, 80 * len(x_labels)),
                height=max(500, 60 * len(y_labels)),
                xaxis=dict(tickangle=-45, title='Heat production (by resource)'),
                yaxis=dict(autorange='reversed', title='Electricity production (by technology)'),
            )

            if plot:
                pio.show(fig)

            fig.write_html(outdir + f"_Raw/{_stem}.html")
            masked_df = pd.DataFrame(mask, index=cross.index, columns=cross.columns)
            self._export_corr_heatmap(
                masked_df, x_labels, y_labels,
                title=_title,
                zmin=-1, zmax=1,
                out_dir=outdir,
                filename_stem=_stem,
                interpolation='nearest',
                text_threshold=text_threshold,
            )

            results[heat_tag] = cross

        return results



    def graph_covariance_electricity(self, ampl_uq_collector=None, year='2050', variable='F_year', normalized=True,
                                     plot=True, corr_method='pearson', min_nonzero_share=0.05,
                                     min_std=1e-6, zero_tol=1e-9):
        """
        Matrice de covariance (ou covariance normalisée) des technologies électriques.

        Parameters
        ----------
        year : str|int
            Année analysée (ex: '2050') ou 'all' pour agréger toutes les années.
        variable : str
            Variable à comparer. Cherche d'abord dans Assets (ex: 'F_year', 'F'),
            puis dans Decision_tracking si absente (ex: 'F_decided_realized_up_to').
        normalized : bool
            True => matrice de corrélation (covariance normalisée, bornée entre -1 et 1).
            False => covariance brute.
        corr_method : str
            Méthode de corrélation pour normalized=True ('pearson', 'spearman', 'kendall').
        min_nonzero_share : float
            Part minimale de scénarios non-nuls pour conserver une technologie.
        min_std : float
            Écart-type minimal pour conserver une technologie.
        zero_tol : float
            Tolérance pour définir une valeur comme nulle.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Assets' not in ampl_uq_collector:
            raise ValueError("'Assets' introuvable dans le collecteur UQ.")

        assets = ampl_uq_collector['Assets'].copy().reset_index()
        source_name = 'Assets'
        if variable in assets.columns:
            source_df = assets
        elif 'Decision_tracking' in ampl_uq_collector:
            decision_tracking = ampl_uq_collector['Decision_tracking'].copy().reset_index()
            if variable in decision_tracking.columns:
                source_df = decision_tracking
                source_name = 'Decision_tracking'
            else:
                raise ValueError(
                    f"Colonne '{variable}' introuvable ni dans 'Assets' ni dans 'Decision_tracking'."
                )
        else:
            raise ValueError(
                f"Colonne '{variable}' introuvable dans 'Assets' et 'Decision_tracking' est absent."
            )

        year_col = 'Years' if 'Years' in source_df.columns else None
        tech_col = 'Technologies' if 'Technologies' in source_df.columns else source_df.columns[1]
        sample_col = 'Sample' if 'Sample' in source_df.columns else None

        if sample_col is None:
            source_df['Sample'] = 0
            sample_col = 'Sample'

        elec_techs = self._get_electricity_technologies()
        source_df = source_df.loc[source_df[tech_col].isin(elec_techs)].copy()
        if source_df.empty:
            raise ValueError(f"Aucune technologie électrique trouvée dans {source_name}.")

        if str(year).lower() != 'all' and year_col is not None:
            year_text = str(year)
            valid_year_keys = {year_text, f'YEAR_{year_text}'}
            source_df = source_df.loc[source_df[year_col].astype(str).isin(valid_year_keys)].copy()
            if source_df.empty:
                raise ValueError(f"Aucune donnée disponible pour l'année {year_text}.")

        grouped = source_df.groupby([sample_col, tech_col], as_index=False)[variable].sum()
        matrix_input = grouped.pivot(index=sample_col, columns=tech_col, values=variable).fillna(0)

        tech_order = [tech for tech in elec_techs if tech in matrix_input.columns]
        if len(tech_order) < 2:
            raise ValueError("Pas assez de technologies électriques avec données pour calculer une matrice.")

        matrix_input = matrix_input.reindex(columns=tech_order, fill_value=0)

        nonzero_share = (matrix_input.abs() > zero_tol).mean(axis=0)
        std_dev = matrix_input.std(axis=0, ddof=0)
        keep_mask = (nonzero_share >= min_nonzero_share) & (std_dev >= min_std)

        kept_techs = [tech for tech in tech_order if bool(keep_mask.get(tech, False))]
        if len(kept_techs) < 2:
            raise ValueError(
                "Pas assez de technologies électriques après filtrage "
                f"(min_nonzero_share={min_nonzero_share}, min_std={min_std})."
            )

        matrix_input = matrix_input[kept_techs]
        tech_order = kept_techs

        if normalized:
            matrix = matrix_input.corr(method=corr_method).fillna(0)
            zmin, zmax = -1, 1
            cb_title = 'Covariance normalisée [-1,1]'
            title_main = "<b>Matrice de covariance normalisée (corrélation)</b>"
        else:
            matrix = matrix_input.cov().fillna(0)
            max_abs = float(np.nanmax(np.abs(matrix.values))) if matrix.size > 0 else 1.0
            if max_abs <= 0:
                max_abs = 1.0
            zmin, zmax = -max_abs, max_abs
            cb_title = 'Covariance'
            title_main = "<b>Matrice de covariance</b>"

        meaning = self.dict_meaning()
        labels = [meaning.get(tech, tech) for tech in tech_order]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                text=np.round(matrix.values, 2),
                texttemplate='%{text}',
                textfont=dict(color='black', size=11),
                x=labels,
                y=labels,
                zmin=zmin,
                zmax=zmax,
                colorscale='RdBu_r',
                colorbar=dict(title=cb_title),
                hovertemplate='X: %{x}<br>Y: %{y}<br>Valeur: %{z:.2f}<extra></extra>'
            )
        )

        year_label = str(year)
        title = f"{title_main}<br>Technologies électriques - année {year_label}"
        if str(year).lower() == 'all':
            title = f"{title_main}<br>Technologies électriques - toutes années"
        elif year_col is None:
            title = f"{title_main}<br>Technologies électriques - toutes périodes ({source_name})"

        title += f"<br><sup>Filtre: part non-nulle ≥ {min_nonzero_share:.0%}, écart-type ≥ {min_std:g}</sup>"

        fig.update_layout(
            title=title,
            template='simple_white',
            width=1100,
            height=900,
            xaxis=dict(title='Technologies', tickangle=-45),
            yaxis=dict(title='Technologies', autorange='reversed')
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Covariance/")):
            Path(self.outdir + "Covariance/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Covariance/_Raw/")):
            Path(self.outdir + "Covariance/_Raw/").mkdir(parents=True, exist_ok=True)

        suffix = 'normalized' if normalized else 'raw'
        year_suffix = str(year).replace('/', '_')
        fig.write_html(self.outdir + f"Covariance/_Raw/Covariance_electricity_{variable}_{year_suffix}_{suffix}.html")
        fig.write_image(self.outdir + f"Covariance/Covariance_electricity_{variable}_{year_suffix}_{suffix}.pdf", width=1100, height=900)

        diagnostics = pd.DataFrame({
            'Technology': list(nonzero_share.index),
            'NonzeroShare': [float(nonzero_share[t]) for t in nonzero_share.index],
            'StdDev': [float(std_dev[t]) for t in std_dev.index],
            'Kept': [bool(keep_mask[t]) for t in nonzero_share.index]
        })
        diagnostics.to_csv(self.outdir + f"Covariance/Covariance_electricity_{variable}_{year_suffix}_diagnostics.csv", index=False)

        matrix_out = matrix.copy()
        matrix_out.index = tech_order
        matrix_out.columns = tech_order
        matrix_out.to_csv(self.outdir + f"Covariance/Covariance_electricity_{variable}_{year_suffix}_{suffix}.csv")

        return matrix_out


    def graph_decision_correlation_period(self, ampl_uq_collector=None,
                                          year_start=2030, year_end=2039,
                                          corr_method='pearson', plot=True,
                                          text_threshold=0.2,
                                          exclude_technologies=None, include_ccgt=False):
        """
        Matrice de corrélation de la capacité décidée (F_decided_realized_up_to)
        entre technologies de production électrique, sur une tranche d'années.

        Pour chaque scénario, on somme F_decided_realized_up_to sur les phases
        dont l'année de début est comprise dans [year_start, year_end], puis on
        calcule la corrélation entre toutes les paires de technologies.

        Parameters
        ----------
        year_start : int
            Première année de la tranche (incluse). Ex: 2030.
        year_end : int
            Dernière année de la tranche (incluse). Ex: 2039.
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        plot : bool
            Afficher le graphe interactif Plotly.
        text_threshold : float
            Les lignes/colonnes dont le max |corrélation hors-diagonale| est
            inférieur à cette valeur sont supprimées. Les valeurs en dessous
            ne sont pas affichées.
        exclude_technologies : list[str] | None
            Technologies à exclure (en plus des exclusions par défaut).
        include_ccgt : bool
            Si True, inclure les technologies CCGT.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in dt.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col  = 'Phases'       if 'Phases'       in dt.columns else dt.columns[0]
        tech_col   = 'Technologies' if 'Technologies' in dt.columns else dt.columns[1]
        sample_col = 'Sample'
        if sample_col not in dt.columns:
            dt[sample_col] = 0

        # --- filtrer les technologies électriques ---
        elec_techs = self._get_electricity_technologies()
        _default_exclude = {'HYDRO_RIVER', 'COAL_IGCC', 'COAL_US', 'CCGT_AMMONIA'}
        _user_exclude = set(exclude_technologies) if exclude_technologies else set()
        if not include_ccgt:
            _user_exclude |= {t for t in elec_techs if 'CCGT' in t.upper()}
        elec_techs = [t for t in elec_techs if t not in _default_exclude | _user_exclude]
        dt = dt.loc[dt[tech_col].isin(elec_techs)].copy()
        if dt.empty:
            raise ValueError("Aucune donnée de décision trouvée pour les technologies électriques.")

        # --- filtrer les phases dans la tranche d'années ---
        def _phase_start(phase_str):
            nums = [int(x) for x in re.findall(r'\d+', str(phase_str))]
            return nums[0] if nums else None

        dt['_phase_start'] = dt[phase_col].map(_phase_start)
        dt = dt.loc[
            dt['_phase_start'].notna() &
            (dt['_phase_start'] >= int(year_start)) &
            (dt['_phase_start'] <= int(year_end))
        ].copy()
        if dt.empty:
            raise ValueError(
                f"Aucune phase trouvée dans la tranche [{year_start}, {year_end}]. "
                "Vérifiez year_start / year_end par rapport aux phases disponibles."
            )

        # --- agréger par (sample, technologie) ---
        grouped = (
            dt.groupby([sample_col, tech_col], as_index=False)['F_decided_realized_up_to']
            .sum()
        )
        pivot = (
            grouped
            .pivot(index=sample_col, columns=tech_col, values='F_decided_realized_up_to')
            .fillna(0)
        )

        # supprimer les colonnes à variance nulle
        pivot = pivot.loc[:, pivot.std(ddof=0) > 1e-9]
        if pivot.shape[1] < 2:
            raise ValueError("Pas assez de technologies avec données pour calculer une corrélation.")

        corr_matrix = pivot.corr(method=corr_method)

        # supprimer les lignes/colonnes dont aucune corrélation hors-diag ne dépasse le seuil
        off_diag = corr_matrix.copy()
        np.fill_diagonal(off_diag.values, 0.0)
        keep = off_diag.abs().ge(text_threshold).any(axis=1)
        corr_matrix = corr_matrix.loc[keep, keep]

        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        labels = list(corr_matrix.columns)
        labels_display = [meaning.get(lbl, self._fmt_tech_label(lbl)) for lbl in labels]

        # masque triangulaire inférieur + seuil
        mask = corr_matrix.values.copy()
        n = mask.shape[0]
        mask[np.triu_indices(n, k=1)] = np.nan
        mask[np.abs(mask) < text_threshold] = np.nan
        text_mask = np.where(np.isnan(mask), '', np.round(mask, 2).astype(str))

        period_label = f"{year_start}–{year_end}"
        _title = f"Decided capacity correlation ({period_label})"
        _stem  = f"Decision_corr_{year_start}_{year_end}_{corr_method}"

        fig = go.Figure(data=go.Heatmap(
            z=mask,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=labels_display,
            y=labels_display,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x} × %{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=_title,
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=200, r=80, t=90, b=200),
            width=max(800, 60 * len(labels_display)),
            height=max(800, 60 * len(labels_display)),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "DecisionCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/{_stem}.html")

        masked_df = pd.DataFrame(mask, index=corr_matrix.index, columns=corr_matrix.columns)
        self._export_corr_heatmap(
            masked_df, labels_display, labels_display,
            title=_title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem=_stem,
            interpolation='nearest',
            text_threshold=text_threshold,
        )

        corr_matrix.to_csv(outdir + f"{_stem}.csv")

        return corr_matrix


    def graph_installed_capacity_correlation_period(self, ampl_uq_collector=None,
                                                    year_start=2030, year_end=2039,
                                                    variable='F',
                                                    corr_method='pearson', plot=True,
                                                    text_threshold=0.3,
                                                    exclude_technologies=None, include_ccgt=False):
        """
        Matrice de corrélation de la capacité installée (Assets) entre technologies
        de production électrique, sur une tranche d'années.

        Pour chaque scénario, on somme `variable` sur les années comprises dans
        [year_start, year_end], puis on calcule la corrélation entre toutes les
        paires de technologies.

        Parameters
        ----------
        year_start : int
            Première année de la tranche (incluse). Ex: 2030.
        year_end : int
            Dernière année de la tranche (incluse). Ex: 2039.
        variable : str
            Colonne de Assets à utiliser (ex: 'F', 'F_year').
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        plot : bool
            Afficher le graphe interactif Plotly.
        text_threshold : float
            Les lignes/colonnes dont le max |corrélation hors-diagonale| est
            inférieur à cette valeur sont supprimées. Les valeurs en dessous
            ne sont pas affichées.
        exclude_technologies : list[str] | None
            Technologies à exclure (en plus des exclusions par défaut).
        include_ccgt : bool
            Si True, inclure les technologies CCGT.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Assets' not in ampl_uq_collector:
            raise ValueError("'Assets' introuvable dans le collecteur UQ.")

        assets = ampl_uq_collector['Assets'].copy().reset_index()
        if variable not in assets.columns:
            raise ValueError(f"Colonne '{variable}' introuvable dans 'Assets'.")

        year_col   = 'Years'       if 'Years'       in assets.columns else None
        tech_col   = 'Technologies' if 'Technologies' in assets.columns else assets.columns[1]
        sample_col = 'Sample'
        if sample_col not in assets.columns:
            assets[sample_col] = 0

        # --- filtrer les technologies électriques ---
        elec_techs = self._get_electricity_technologies()
        _default_exclude = {'HYDRO_RIVER', 'COAL_IGCC', 'COAL_US', 'CCGT_AMMONIA'}
        _user_exclude = set(exclude_technologies) if exclude_technologies else set()
        if not include_ccgt:
            _user_exclude |= {t for t in elec_techs if 'CCGT' in t.upper()}
        elec_techs = [t for t in elec_techs if t not in _default_exclude | _user_exclude]
        assets = assets.loc[assets[tech_col].isin(elec_techs)].copy()
        if assets.empty:
            raise ValueError("Aucune technologie électrique trouvée dans Assets.")

        # --- filtrer la tranche d'années ---
        if year_col is not None:
            def _to_int(v):
                nums = re.findall(r'\d+', str(v))
                return int(nums[0]) if nums else None

            assets['_yr'] = assets[year_col].astype(str).str.replace('YEAR_', '', regex=False).apply(_to_int)
            assets = assets[(assets['_yr'] >= int(year_start)) & (assets['_yr'] <= int(year_end))].copy()
            if assets.empty:
                raise ValueError(
                    f"Aucune donnée entre {year_start} et {year_end} dans Assets."
                )

        # --- agréger par (sample, technologie) ---
        pivot = (
            assets.groupby([sample_col, tech_col], as_index=False)[variable]
            .sum()
            .pivot(index=sample_col, columns=tech_col, values=variable)
            .fillna(0)
        )

        # supprimer les colonnes à variance nulle
        pivot = pivot.loc[:, pivot.std(ddof=0) > 1e-9]
        if pivot.shape[1] < 2:
            raise ValueError("Pas assez de technologies avec données pour calculer une corrélation.")

        corr_matrix = pivot.corr(method=corr_method)

        # supprimer les lignes/colonnes dont aucune corrélation hors-diag ne dépasse le seuil
        off_diag = corr_matrix.copy()
        np.fill_diagonal(off_diag.values, 0.0)
        keep = off_diag.abs().ge(text_threshold).any(axis=1)
        corr_matrix = corr_matrix.loc[keep, keep]

        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        labels = list(corr_matrix.columns)
        labels_display = [meaning.get(lbl, self._fmt_tech_label(lbl)) for lbl in labels]

        # masque triangulaire inférieur + seuil
        mask = corr_matrix.values.copy()
        n = mask.shape[0]
        mask[np.triu_indices(n, k=1)] = np.nan
        mask[np.abs(mask) < text_threshold] = np.nan
        text_mask = np.where(np.isnan(mask), '', np.round(mask, 2).astype(str))

        _title = f"Installed capacity correlation ({year_start})"
        _stem  = f"Installed_cap_corr_{year_start}_{year_end}_{corr_method}"

        fig = go.Figure(data=go.Heatmap(
            z=mask,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=labels_display,
            y=labels_display,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x} × %{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=_title,
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=200, r=80, t=90, b=200),
            width=max(800, 60 * len(labels_display)),
            height=max(800, 60 * len(labels_display)),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "InstalledCapCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/{_stem}.html")

        masked_df = pd.DataFrame(mask, index=corr_matrix.index, columns=corr_matrix.columns)
        self._export_corr_heatmap(
            masked_df, labels_display, labels_display,
            title=_title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem=_stem,
            interpolation='nearest',
            text_threshold=text_threshold,
        )

        corr_matrix.to_csv(outdir + f"{_stem}.csv")

        return corr_matrix


    def graph_elec_production_correlation_period(self, ampl_uq_collector=None,
                                                 year_start=2030, year_end=2039,
                                                 corr_method='pearson', plot=True,
                                                 text_threshold=0.1,
                                                 elec_threshold_share=0.01):
        """
        Matrice de corrélation de la production d'électricité entre technologies,
        sur une tranche d'années.

        Source : Year_balance, colonne ELECTRICITY (valeurs positives = production).
        Pour chaque scénario, on somme la production sur [year_start, year_end]
        par technologie, puis on calcule la corrélation entre toutes les paires.

        Parameters
        ----------
        year_start : int
            Première année de la tranche (incluse).
        year_end : int
            Dernière année de la tranche (incluse).
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        plot : bool
            Afficher le graphe interactif Plotly.
        text_threshold : float
            Les lignes/colonnes dont le max |corrélation hors-diagonale| est
            inférieur à cette valeur sont supprimées. Les valeurs en dessous
            ne sont pas affichées.
        elec_threshold_share : float
            Part minimale de la production totale pour garder une technologie
            (filtre les technologies anecdotiques). Défaut : 1 %.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        year_col   = 'Years'    if 'Years'    in yb.columns else yb.columns[0]
        elem_col   = 'Elements' if 'Elements' in yb.columns else yb.columns[1]
        sample_col = 'Sample'
        if sample_col not in yb.columns:
            yb[sample_col] = 0

        if 'ELECTRICITY' not in yb.columns:
            raise ValueError("Colonne 'ELECTRICITY' introuvable dans Year_balance.")

        # --- filtrer la tranche d'années ---
        yb['_yr'] = (
            yb[year_col].astype(str)
            .str.replace('YEAR_', '', regex=False)
            .apply(lambda v: int(re.findall(r'\d+', v)[0]) if re.findall(r'\d+', v) else None)
        )
        yb = yb[(yb['_yr'] >= int(year_start)) & (yb['_yr'] <= int(year_end))].copy()
        if yb.empty:
            raise ValueError(f"Aucune donnée entre {year_start} et {year_end} dans Year_balance.")

        # --- garder uniquement les technologies qui produisent de l'électricité ---
        yb_prod = yb[yb['ELECTRICITY'] > 0].copy()
        elec_total = yb_prod.groupby(elem_col)['ELECTRICITY'].sum()
        threshold = elec_total.sum() * float(elec_threshold_share)
        producers = elec_total[elec_total >= threshold].index.tolist()
        if len(producers) < 2:
            raise ValueError(
                "Pas assez de technologies productrices d'électricité après filtrage. "
                f"Essayez de réduire elec_threshold_share (actuellement {elec_threshold_share})."
            )

        # --- construire les séries par (sample, technologie) ---
        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        records = {}
        for elem in producers:
            sub = yb_prod[yb_prod[elem_col] == elem]
            label = meaning.get(str(elem), self._fmt_tech_label(str(elem)))
            records[label] = sub.groupby(sample_col)['ELECTRICITY'].sum()

        pivot = pd.DataFrame(records).fillna(0)
        means = pivot.mean()
        cv = pivot.std(ddof=0) / means.replace(0, np.nan)
        pivot = pivot.loc[:, cv.fillna(0) > 0.001]
        if pivot.shape[1] < 2:
            raise ValueError("Pas assez de technologies avec variance non nulle.")

        corr_matrix = pivot.corr(method=corr_method)

        # supprimer les lignes/colonnes dont aucune corrélation hors-diag ne dépasse le seuil
        off_diag = corr_matrix.copy()
        np.fill_diagonal(off_diag.values, 0.0)
        keep = off_diag.abs().ge(text_threshold).any(axis=1)
        corr_matrix = corr_matrix.loc[keep, keep]

        labels_display = list(corr_matrix.columns)

        # masque triangulaire inférieur + seuil
        mask = corr_matrix.values.copy()
        n = mask.shape[0]
        mask[np.triu_indices(n, k=1)] = np.nan
        mask[np.abs(mask) < text_threshold] = np.nan
        text_mask = np.where(np.isnan(mask), '', np.round(mask, 2).astype(str))

        _title = f"Electricity production correlation ({year_start})"
        _stem  = f"Elec_prod_corr_{year_start}_{year_end}_{corr_method}"

        fig = go.Figure(data=go.Heatmap(
            z=mask,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=labels_display,
            y=labels_display,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x} × %{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=_title,
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=200, r=80, t=90, b=200),
            width=max(800, 60 * len(labels_display)),
            height=max(800, 60 * len(labels_display)),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "ElecProductionCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/{_stem}.html")

        masked_df = pd.DataFrame(mask, index=corr_matrix.index, columns=corr_matrix.columns)
        self._export_corr_heatmap(
            masked_df, labels_display, labels_display,
            title=_title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem=_stem,
            interpolation='nearest',
            text_threshold=text_threshold,
        )

        corr_matrix.to_csv(outdir + f"{_stem}.csv")

        return corr_matrix

    def graph_layer_production_correlation(self, layer, ampl_uq_collector=None,
                                            year_start=2045, year_end=2045,
                                            corr_method='pearson', plot=True,
                                            text_threshold=0.1, min_share=0.01):
        """
        Matrice de corrélation de la production entre technologies pour un layer donné.

        Parameters
        ----------
        layer : str
            Colonne Year_balance à analyser (ex: 'METHANOL', 'H2', 'HEAT_HIGH_T').
        year_start / year_end : int
            Tranche d'années (incluses).
        min_share : float
            Part minimale de la production totale pour garder une technologie.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()
        if layer not in yb.columns:
            raise ValueError(f"Layer '{layer}' introuvable dans Year_balance. "
                             f"Disponibles : {[c for c in yb.columns if c not in ('Sample',)][:10]}")

        year_col   = 'Years'    if 'Years'    in yb.columns else yb.columns[0]
        elem_col   = 'Elements' if 'Elements' in yb.columns else yb.columns[1]
        sample_col = 'Sample'
        if sample_col not in yb.columns:
            yb[sample_col] = 0

        yb['_yr'] = (yb[year_col].astype(str)
                     .str.replace('YEAR_', '', regex=False)
                     .apply(lambda v: int(re.findall(r'\d+', v)[0]) if re.findall(r'\d+', v) else None))
        yb = yb[(yb['_yr'] >= int(year_start)) & (yb['_yr'] <= int(year_end))].copy()
        if yb.empty:
            raise ValueError(f"Aucune donnée entre {year_start} et {year_end}.")

        yb_prod = yb[yb[layer] > 0].copy()
        total   = yb_prod.groupby(elem_col)[layer].sum()
        threshold = total.sum() * float(min_share)
        producers = total[total >= threshold].index.tolist()
        if len(producers) < 2:
            raise ValueError("Pas assez de producteurs — baisse min_share.")

        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        records = {}
        for elem in producers:
            sub   = yb_prod[yb_prod[elem_col] == elem]
            label = meaning.get(str(elem), self._fmt_tech_label(str(elem)))
            records[label] = sub.groupby(sample_col)[layer].sum()

        pivot = pd.DataFrame(records).fillna(0)
        means = pivot.mean()
        cv    = pivot.std(ddof=0) / means.replace(0, np.nan)
        pivot = pivot.loc[:, cv.fillna(0) > 0.001]
        if pivot.shape[1] < 2:
            raise ValueError("Pas assez de technologies avec variance non nulle.")

        corr_matrix = pivot.corr(method=corr_method)
        off_diag = corr_matrix.copy()
        np.fill_diagonal(off_diag.values, 0.0)
        keep = off_diag.abs().ge(text_threshold).any(axis=1)
        corr_matrix = corr_matrix.loc[keep, keep]

        labels = list(corr_matrix.columns)
        mask   = corr_matrix.values.copy()
        n = mask.shape[0]
        mask[np.triu_indices(n, k=1)] = np.nan
        mask[np.abs(mask) < text_threshold] = np.nan

        _title = f"{layer} production correlation ({year_start})"
        _stem  = f"{layer}_prod_corr_{year_start}_{year_end}_{corr_method}"

        fig = go.Figure(data=go.Heatmap(
            z=mask,
            text=np.where(np.isnan(mask), '', np.round(mask, 2).astype(str)),
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=labels, y=labels,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x} × %{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))
        fig.update_layout(
            title=dict(text=_title, x=0.5, xanchor='center',
                       font=dict(family='Raleway', size=28)),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=200, r=80, t=90, b=200),
            width=max(800, 60 * len(labels)),
            height=max(800, 60 * len(labels)),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + f"{layer}ProductionCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/{_stem}.html")

        masked_df = pd.DataFrame(mask, index=corr_matrix.index, columns=corr_matrix.columns)
        self._export_corr_heatmap(
            masked_df, labels, labels,
            title=_title, zmin=-1, zmax=1,
            out_dir=outdir, filename_stem=_stem,
            interpolation='nearest', text_threshold=text_threshold,
        )
        corr_matrix.to_csv(outdir + f"{_stem}.csv")
        return corr_matrix

    def graph_heat_high_t_production_correlation(self, ampl_uq_collector=None,
                                                  year_start=2040, year_end=2050,
                                                  corr_method='pearson', plot=True,
                                                  text_threshold=0.3,
                                                  prod_threshold_share=0.01):
        """
        Matrice de corrélation de la production de chaleur haute température
        entre technologies, sur une tranche d'années.

        Source : Year_balance, colonne HEAT_HIGH_T (valeurs positives = production).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        year_col   = 'Years'    if 'Years'    in yb.columns else yb.columns[0]
        elem_col   = 'Elements' if 'Elements' in yb.columns else yb.columns[1]
        sample_col = 'Sample'
        if sample_col not in yb.columns:
            yb[sample_col] = 0

        if 'HEAT_HIGH_T' not in yb.columns:
            raise ValueError("Colonne 'HEAT_HIGH_T' introuvable dans Year_balance.")

        yb['_yr'] = (
            yb[year_col].astype(str)
            .str.replace('YEAR_', '', regex=False)
            .apply(lambda v: int(re.findall(r'\d+', v)[0]) if re.findall(r'\d+', v) else None)
        )
        yb = yb[(yb['_yr'] >= int(year_start)) & (yb['_yr'] <= int(year_end))].copy()
        if yb.empty:
            raise ValueError(f"Aucune donnée entre {year_start} et {year_end} dans Year_balance.")

        yb_prod = yb[yb['HEAT_HIGH_T'] > 0].copy()
        heat_total = yb_prod.groupby(elem_col)['HEAT_HIGH_T'].sum()
        threshold = heat_total.sum() * float(prod_threshold_share)
        producers = heat_total[heat_total >= threshold].index.tolist()
        if len(producers) < 2:
            raise ValueError(
                "Pas assez de technologies productrices après filtrage. "
                f"Essayez de réduire prod_threshold_share (actuellement {prod_threshold_share})."
            )

        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        records = {}
        for elem in producers:
            sub = yb_prod[yb_prod[elem_col] == elem]
            label = meaning.get(str(elem), self._fmt_tech_label(str(elem)))
            records[label] = sub.groupby(sample_col)['HEAT_HIGH_T'].sum()

        pivot = pd.DataFrame(records).fillna(0)
        pivot = pivot.loc[:, pivot.std(ddof=0) > 1e-9]
        if pivot.shape[1] < 2:
            raise ValueError("Pas assez de technologies avec variance non nulle.")

        corr_matrix = pivot.corr(method=corr_method)

        off_diag = corr_matrix.copy()
        np.fill_diagonal(off_diag.values, 0.0)
        keep = off_diag.abs().ge(text_threshold).any(axis=1)
        corr_matrix = corr_matrix.loc[keep, keep]

        labels_display = list(corr_matrix.columns)

        mask = corr_matrix.values.copy()
        n = mask.shape[0]
        mask[np.triu_indices(n, k=1)] = np.nan
        mask[np.abs(mask) < text_threshold] = np.nan
        text_mask = np.where(np.isnan(mask), '', np.round(mask, 2).astype(str))

        _title = f"Heat High-T production correlation ({year_start})"
        _stem  = f"HeatHighT_prod_corr_{year_start}_{year_end}_{corr_method}"

        fig = go.Figure(data=go.Heatmap(
            z=mask,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=labels_display,
            y=labels_display,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x} × %{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=_title,
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=200, r=80, t=90, b=200),
            width=max(800, 60 * len(labels_display)),
            height=max(800, 60 * len(labels_display)),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "HeatHighTCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/{_stem}.html")

        masked_df = pd.DataFrame(mask, index=corr_matrix.index, columns=corr_matrix.columns)
        self._export_corr_heatmap(
            masked_df, labels_display, labels_display,
            title=_title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem=_stem,
            interpolation='nearest',
            text_threshold=text_threshold,
        )

        corr_matrix.to_csv(outdir + f"{_stem}.csv")

        return corr_matrix


    def graph_heat_low_t_production_correlation(self, ampl_uq_collector=None,
                                                year_start=2040, year_end=2050,
                                                corr_method='pearson', plot=True,
                                                text_threshold=0.3,
                                                prod_threshold_share=0.01):
        """
        Matrice de corrélation de la production de chaleur basse température
        entre technologies, sur une tranche d'années.

        Source : Year_balance, colonnes HEAT_LOW_T_DHN + HEAT_LOW_T_DECEN
        (valeurs positives = production, sommées en une seule colonne HEAT_LOW_T).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        year_col   = 'Years'    if 'Years'    in yb.columns else yb.columns[0]
        elem_col   = 'Elements' if 'Elements' in yb.columns else yb.columns[1]
        sample_col = 'Sample'
        if sample_col not in yb.columns:
            yb[sample_col] = 0

        lt_cols = [c for c in yb.columns if 'HEAT_LOW_T_DHN' in str(c)] + \
                  [c for c in yb.columns if 'HEAT_LOW_T_DECEN' in str(c)]
        if not lt_cols:
            raise ValueError("Aucune colonne HEAT_LOW_T_DHN / HEAT_LOW_T_DECEN dans Year_balance.")
        yb['HEAT_LOW_T'] = yb[lt_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

        yb['_yr'] = (
            yb[year_col].astype(str)
            .str.replace('YEAR_', '', regex=False)
            .apply(lambda v: int(re.findall(r'\d+', v)[0]) if re.findall(r'\d+', v) else None)
        )
        yb = yb[(yb['_yr'] >= int(year_start)) & (yb['_yr'] <= int(year_end))].copy()
        if yb.empty:
            raise ValueError(f"Aucune donnée entre {year_start} et {year_end} dans Year_balance.")

        yb_prod = yb[yb['HEAT_LOW_T'] > 0].copy()
        heat_total = yb_prod.groupby(elem_col)['HEAT_LOW_T'].sum()
        threshold = heat_total.sum() * float(prod_threshold_share)
        producers = heat_total[heat_total >= threshold].index.tolist()
        if len(producers) < 2:
            raise ValueError(
                "Pas assez de technologies productrices après filtrage. "
                f"Essayez de réduire prod_threshold_share (actuellement {prod_threshold_share})."
            )

        meaning = self.dict_meaning() if hasattr(self, 'dict_meaning') else {}
        records = {}
        for elem in producers:
            sub = yb_prod[yb_prod[elem_col] == elem]
            label = meaning.get(str(elem), self._fmt_tech_label(str(elem)))
            records[label] = sub.groupby(sample_col)['HEAT_LOW_T'].sum()

        pivot = pd.DataFrame(records).fillna(0)
        pivot = pivot.loc[:, pivot.std(ddof=0) > 1e-9]
        if pivot.shape[1] < 2:
            raise ValueError("Pas assez de technologies avec variance non nulle.")

        corr_matrix = pivot.corr(method=corr_method)

        off_diag = corr_matrix.copy()
        np.fill_diagonal(off_diag.values, 0.0)
        keep = off_diag.abs().ge(text_threshold).any(axis=1)
        corr_matrix = corr_matrix.loc[keep, keep]

        labels_display = list(corr_matrix.columns)

        mask = corr_matrix.values.copy()
        n = mask.shape[0]
        mask[np.triu_indices(n, k=1)] = np.nan
        mask[np.abs(mask) < text_threshold] = np.nan
        text_mask = np.where(np.isnan(mask), '', np.round(mask, 2).astype(str))

        period_label = f"{year_start}–{year_end}"
        _title = f"Heat Low-T production correlation ({period_label})"
        _stem  = f"HeatLowT_prod_corr_{year_start}_{year_end}_{corr_method}"

        fig = go.Figure(data=go.Heatmap(
            z=mask,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=labels_display,
            y=labels_display,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x} × %{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=_title,
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=200, r=80, t=90, b=200),
            width=max(800, 60 * len(labels_display)),
            height=max(800, 60 * len(labels_display)),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        outdir = self.outdir + "HeatLowTCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/{_stem}.html")

        masked_df = pd.DataFrame(mask, index=corr_matrix.index, columns=corr_matrix.columns)
        self._export_corr_heatmap(
            masked_df, labels_display, labels_display,
            title=_title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem=_stem,
            interpolation='nearest',
            text_threshold=text_threshold,
        )

        corr_matrix.to_csv(outdir + f"{_stem}.csv")

        return corr_matrix


    def graph_covariance_price_construction_time(self, ampl_uq_collector=None, price_col=None,
                                                 construction_cols=None, construction_prefix='cp_',
                                                 normalized=True, corr_method='pearson', plot=True):
        """
        Matrice de covariance entre le prix et les temps de construction.

        Les temps de construction sont exponentiés avant calcul (hypothèse log-normale).

        Parameters
        ----------
        ampl_uq_collector : dict | None
            Collecteur UQ. Si None, utilise self.ampl_uq_collector.
        price_col : str | None
            Nom de la colonne prix dans Samples. Si None, essaie self.objective,
            puis des fallback usuels ('cost', 'price', 'total_cost').
        construction_cols : list[str] | None
            Colonnes de temps de construction dans Samples. Si None, détecte
            automatiquement via le préfixe construction_prefix puis des motifs
            contenant 'constr'/'construction'.
        construction_prefix : str
            Préfixe prioritaire de détection des colonnes de construction.
        normalized : bool
            True => matrice de corrélation ; False => matrice de covariance.
        corr_method : str
            Méthode de corrélation si normalized=True ('pearson', 'spearman', 'kendall').
        plot : bool
            Si True, affiche la figure.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)

        if price_col is None:
            candidates = []
            if isinstance(self.objective, str):
                candidates.append(self.objective)
            candidates.extend(['cost', 'price', 'total_cost'])
            price_col = next((c for c in candidates if c in samples.columns), None)

        if price_col is None or price_col not in samples.columns:
            raise ValueError(
                f"Colonne prix introuvable. Colonnes disponibles: {list(samples.columns)}"
            )

        if construction_cols is None:
            cols_by_prefix = [
                c for c in samples.columns
                if isinstance(c, str) and c.startswith(construction_prefix)
            ]
            if len(cols_by_prefix) > 0:
                construction_cols = cols_by_prefix
            else:
                construction_cols = [
                    c for c in samples.columns
                    if isinstance(c, str) and re.search(r'constr|construction|t_constr', c, flags=re.IGNORECASE)
                ]

        construction_cols = [c for c in construction_cols if c in samples.columns and c != price_col]
        if len(construction_cols) == 0:
            raise ValueError(
                "Aucune colonne de temps de construction détectée dans Samples. "
                "Passez construction_cols explicitement si nécessaire."
            )

        cols_for_cov = [price_col] + construction_cols
        data = samples[cols_for_cov].apply(pd.to_numeric, errors='coerce')

        # Les colonnes de construction sont log-espace: on revient en espace réel qu'on arrondi
        data[construction_cols] = np.ceil(np.exp(data[construction_cols]))
        data.dropna(how='any', inplace=True)

        if data.shape[0] < 2:
            raise ValueError("Pas assez de lignes valides pour calculer la covariance.")

        if normalized:
            # Conserver les correlations reelles avec l'echelle couleur complete [-1, 1].
            matrix = data.corr(method=corr_method).fillna(0)
            zmin, zmax = -1, 1
            cb_title = ''
            title = "<b>Cost and commissioning time correlation</b>"
            suffix = 'normalized'
        else:
            matrix = data.cov().fillna(0)
            max_abs = float(np.nanmax(np.abs(matrix.values))) if matrix.size > 0 else 1.0
            if max_abs <= 0:
                max_abs = 1.0
            zmin, zmax = -max_abs, max_abs
            cb_title = 'Covariance'
            title = "<b>Cost and commissioning time correlation</b>"
            suffix = 'raw'

        row_values = matrix.loc[price_col, construction_cols]
        row_matrix = pd.DataFrame([row_values.values], index=[price_col], columns=construction_cols)
        row_matrix = row_matrix.sort_values(by=price_col, axis=1)

        x_labels = [self.uncert_param_meaning.get(c, c) for c in row_matrix.columns]
        y_labels = [self.uncert_param_meaning.get(price_col, price_col)]

        fig = go.Figure(
            data=go.Heatmap(
                z=row_matrix.values,
                text=np.round(row_matrix.values, 3),
                texttemplate='%{text}',
                textfont=dict(color='black', size=10),
                zsmooth='best',
                x=x_labels,
                y=y_labels,
                zmin=zmin,
                zmax=zmax,
                colorscale='RdBu_r',
                colorbar=dict(title=cb_title, tickvals=[-1,0, 1], ticktext=['-1', '', '1']),
                hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.4f}<extra></extra>'
            )
        )

        title_parts = str(title).split('<br>', 1)
        title_main = re.sub(r'</?b>', '', title_parts[0]).strip()
        title_left = title_parts[1].strip() if len(title_parts) > 1 else None

        fig.update_layout(
            title=dict(
                text=title_main,
                x=0.5,
                xanchor='center',
                font=dict(family="Raleway", size=28)
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=120, r=80, t=90, b=120),
            width=750,
            height=280,
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed')
        )

        if title_left:
            fig.add_annotation(
                xref='paper',
                yref='paper',
                x=0,
                y=1.02,
                text=title_left,
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=18, color='rgb(90,90,90)')
            )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Covariance_PriceConstruction/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        fig.write_html(out_dir + f"_Raw/Covariance_price_construction_{suffix}.html")
        self._export_corr_heatmap(
            row_matrix, x_labels, y_labels,
            title="Cost and commissioning time correlation",
            zmin=zmin, zmax=zmax,
            out_dir=out_dir,
            filename_stem=f"Covariance_price_construction_{suffix}"
        )

        matrix_out = row_matrix.copy()
        matrix_out.to_csv(out_dir + f"Covariance_price_construction_{suffix}.csv")

        diagnostics = pd.DataFrame({
            'Variable': [price_col] + construction_cols,
            'DisplayLabel': y_labels + x_labels,
            'Transform': ['none'] + ['exp'] * len(construction_cols)
        })
        diagnostics.to_csv(out_dir + "Covariance_price_construction_columns.csv", index=False)

        return matrix_out

    def graph_corr_capex_commissioning(self, ampl_uq_collector=None,
                                        year_start=None, year_end=None,
                                        normalized=True, corr_method='pearson',
                                        construction_prefix='cp_',
                                        construction_cols=None,
                                        plot=True):
        """
        Corrélation entre le CAPEX total (somme de C_inv_phase_tech sur toutes les phases
        et technologies) et les temps de commissionnement (cp_*).
        Même format que graph_covariance_price_construction_time : une ligne de heatmap.

        Parameters
        ----------
        year_start / year_end : int | None
            Filtre sur les phases (année de début). Si None, toutes les phases.
        normalized : bool
            True → corrélation de Pearson/Spearman ; False → covariance brute.
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        construction_prefix : str
            Préfixe des colonnes cp_* dans Samples.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_inv_phase_tech' not in ampl_uq_collector:
            raise ValueError("'C_inv_phase_tech' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        # ── CAPEX total par sample ────────────────────────────────────────────────
        inv = ampl_uq_collector['C_inv_phase_tech'].copy().reset_index()
        phase_col = 'Phases' if 'Phases' in inv.columns else inv.columns[0]
        inv['_year'] = inv[phase_col].astype(str).str.split('_').str[0]
        inv['C_inv_phase_tech'] = pd.to_numeric(inv['C_inv_phase_tech'], errors='coerce').fillna(0)

        if year_start is not None:
            inv = inv[inv['_year'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            inv = inv[inv['_year'].apply(_year_key) <= int(year_end)]

        capex_total = inv.groupby('Sample')['C_inv_phase_tech'].sum() / 1000.0
        capex_total.name = 'C_inv_total_bEUR'

        # ── Temps de commissionnement ─────────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)

        if construction_cols is None:
            construction_cols = [c for c in samples.columns
                                  if isinstance(c, str) and c.startswith(construction_prefix)]
            if not construction_cols:
                construction_cols = [c for c in samples.columns
                                      if isinstance(c, str)
                                      and re.search(r'constr|construction|t_constr', c, re.IGNORECASE)]

        construction_cols = [c for c in construction_cols if c in samples.columns]
        if not construction_cols:
            raise ValueError("Aucune colonne de temps de commissionnement détectée dans Samples.")

        cp_df = samples[['Sample'] + construction_cols].copy()
        cp_df[construction_cols] = np.ceil(np.exp(
            cp_df[construction_cols].apply(pd.to_numeric, errors='coerce')
        ))
        cp_df = cp_df.set_index('Sample')

        # ── Fusion & corrélation ──────────────────────────────────────────────────
        merged = capex_total.to_frame().join(cp_df, how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides pour calculer la corrélation.")

        cp_cols = [c for c in construction_cols if c in merged.columns]
        data = merged[['C_inv_total_bEUR'] + cp_cols].apply(pd.to_numeric, errors='coerce')

        if normalized:
            matrix = data.corr(method=corr_method).fillna(0)
            zmin, zmax = -1, 1
            suffix = 'normalized'
        else:
            matrix = data.cov().fillna(0)
            max_abs = float(np.nanmax(np.abs(matrix.values))) or 1.0
            zmin, zmax = -max_abs, max_abs
            suffix = 'raw'

        row_values  = matrix.loc['C_inv_total_bEUR', cp_cols]
        row_matrix  = pd.DataFrame([row_values.values],
                                    index=['C_inv_total'], columns=cp_cols)
        row_matrix  = row_matrix.sort_values(by='C_inv_total', axis=1)

        x_labels = [self.uncert_param_meaning.get(c, c) for c in row_matrix.columns]
        y_labels = ['Total CAPEX']

        fig = go.Figure(data=go.Heatmap(
            z=row_matrix.values,
            text=np.round(row_matrix.values, 3),
            texttemplate='%{text}',
            textfont=dict(color='black', size=10),
            zsmooth='best',
            x=x_labels,
            y=y_labels,
            zmin=zmin, zmax=zmax,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x}<br>Corr: %{z:.4f}<extra></extra>',
        ))
        fig.update_layout(
            title=dict(text='Total CAPEX × commissioning time correlation',
                       x=0.5, xanchor='center',
                       font=dict(family='Raleway', size=28)),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=120, r=80, t=90, b=120),
            width=max(750, 60 * len(x_labels)),
            height=280,
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Covariance_CapexConstruction/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        fig.write_html(out_dir + f"_Raw/capex_construction_{suffix}.html")
        self._export_corr_heatmap(
            row_matrix, x_labels, y_labels,
            title='Total CAPEX × commissioning time correlation',
            zmin=zmin, zmax=zmax,
            out_dir=out_dir,
            filename_stem=f"capex_construction_{suffix}",
        )
        row_matrix.to_csv(out_dir + f"capex_construction_{suffix}.csv")

        return row_matrix

    def graph_corr_inv_sector_commissioning(self, ampl_uq_collector=None,
                                             year_start=None, year_end=None,
                                             corr_method='spearman',
                                             construction_prefix='cp_',
                                             construction_cols=None,
                                             threshold=0.0,
                                             min_spread_beur=0.5,
                                             plot=True):
        """
        Heatmap de corrélation entre le CAPEX total par secteur (somme sur toutes
        les phases 2020-2050) et les temps de commissionnement (cp_*).

        Lignes  : secteurs (ELECTRICITY, HEAT_HIGH_T, HEAT_LOW_T_DHN, …)
        Colonnes: paramètres cp_* (commissioning time)

        Parameters
        ----------
        min_spread_beur : float
            Écart minimum (max - min) en b€ qu'un secteur doit afficher entre
            scénarios pour être inclus. Filtre les secteurs quasi-constants
            dont la corrélation serait numériquement instable (défaut : 0.5 b€).
        threshold : float
            Corrélations dont la valeur absolue est inférieure à ce seuil
            sont masquées (NaN) dans le heatmap (défaut : 0 = tout afficher).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_inv_phase_tech' not in ampl_uq_collector:
            raise ValueError("'C_inv_phase_tech' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")
        if self.category is None:
            raise ValueError("self.category est None — ampl_obj requis pour le mapping secteur.")

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        # ── CAPEX par technologie, tous samples ───────────────────────────────────
        inv = ampl_uq_collector['C_inv_phase_tech'].copy().reset_index()
        phase_col = 'Phases' if 'Phases' in inv.columns else inv.columns[0]
        tech_col  = 'Technologies' if 'Technologies' in inv.columns else inv.columns[1]

        inv['_year'] = inv[phase_col].astype(str).str.split('_').str[0]
        inv['C_inv_phase_tech'] = pd.to_numeric(inv['C_inv_phase_tech'], errors='coerce').fillna(0)

        if year_start is not None:
            inv = inv[inv['_year'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            inv = inv[inv['_year'].apply(_year_key) <= int(year_end)]

        # Mapper technologie → secteur
        inv['Sector'] = inv[tech_col].map(self.category).fillna('OTHER')
        inv['Sector'] = inv['Sector'].map(lambda v: v[0] if isinstance(v, tuple) else v)
        inv['Sector'] = inv['Sector'].astype(str)

        # Agréger par (Sample, Sector), convertir en b€
        capex_sector = (
            inv.groupby(['Sample', 'Sector'])['C_inv_phase_tech']
            .sum()
            .unstack(fill_value=0)
            / 1000.0
        )
        # Supprimer secteurs dont l'écart inter-scénarios est trop faible
        spread = capex_sector.max() - capex_sector.min()
        capex_sector = capex_sector.loc[:, spread >= min_spread_beur]
        if capex_sector.empty:
            raise ValueError(
                f"Aucun secteur ne dépasse le seuil min_spread_beur={min_spread_beur} b€. "
                "Réduire min_spread_beur ou vérifier les données."
            )

        # ── Temps de commissionnement ─────────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)

        if construction_cols is None:
            construction_cols = [c for c in samples.columns
                                  if isinstance(c, str) and c.startswith(construction_prefix)]
            if not construction_cols:
                construction_cols = [c for c in samples.columns
                                      if isinstance(c, str)
                                      and re.search(r'constr|construction|t_constr', c, re.IGNORECASE)]

        construction_cols = [c for c in construction_cols if c in samples.columns]
        if not construction_cols:
            raise ValueError("Aucune colonne de temps de commissionnement dans Samples.")

        cp_df = samples[['Sample'] + construction_cols].copy()
        cp_df[construction_cols] = np.ceil(np.exp(
            cp_df[construction_cols].apply(pd.to_numeric, errors='coerce')
        ))
        cp_df = cp_df.set_index('Sample')

        # ── Fusion & corrélation ──────────────────────────────────────────────────
        merged = capex_sector.join(cp_df, how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios pour calculer la corrélation.")

        sector_cols = list(capex_sector.columns)
        cp_cols = [c for c in construction_cols if c in merged.columns]

        # Corrélation : secteurs (lignes) × cp_params (colonnes)
        corr_matrix = (
            merged[sector_cols + cp_cols]
            .corr(method=corr_method)
            .loc[sector_cols, cp_cols]
        )

        # Ordre des secteurs (même logique que graph_cost_inv_phase_tech_category)
        order_sectors = ['ELECTRICITY', 'HEAT_HIGH_T', 'HEAT_LOW_T_DHN', 'HEAT_LOW_T_DECEN',
                         'MOB_PRIVATE', 'MOB_PUBLIC', 'MOBILITY_FREIGHT',
                         'INFRASTRUCTURE', 'STORAGE', 'HVC', 'METHANOL', 'AMMONIA', 'OTHER']
        row_order = [s for s in order_sectors if s in corr_matrix.index]
        row_order += [s for s in corr_matrix.index if s not in row_order]
        corr_matrix = corr_matrix.loc[row_order]

        # Appliquer le threshold (NaN si en dessous)
        z_vals = corr_matrix.values.copy().astype(float)
        if threshold > 0:
            z_vals[np.abs(z_vals) < threshold] = np.nan

        text_mask = np.where(np.isnan(z_vals), '',
                             np.round(corr_matrix.values, 2).astype(str))

        x_labels = [self.uncert_param_meaning.get(c, c) for c in corr_matrix.columns]
        y_labels = list(corr_matrix.index)

        n_rows = len(y_labels)
        n_cols = len(x_labels)

        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=11),
            x=x_labels,
            y=y_labels,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, -0.5, 0, 0.5, 1],
                          ticktext=['-1', '-0.5', '0', '0.5', '1']),
            hovertemplate='%{x}<br>%{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        yr_label = ''
        if year_start or year_end:
            yr_label = f" ({year_start or 2020}–{year_end or 2050})"

        fig.update_layout(
            title=dict(
                text=f"<b>CAPEX × commissioning time{yr_label}</b>",
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=22),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=13),
            margin=dict(l=180, r=80, t=90, b=180),
            width=max(700, 80 * n_cols),
            height=max(400, 55 * n_rows),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Corr_InvSector_Commission/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        yr_str = f"_{year_start or 2020}_{year_end or 2050}"
        stem = f"corr_inv_sector_commission{yr_str}"
        fig.write_html(out_dir + f"_Raw/{stem}.html")

        self._export_corr_heatmap(
            corr_matrix, x_labels, y_labels,
            title=f"Investment cost × commissioning time{yr_label}",
            zmin=-1, zmax=1,
            out_dir=out_dir,
            filename_stem=stem,
            interpolation='nearest',
            text_threshold=threshold,
            bad_color='white',
        )
        corr_matrix.to_csv(out_dir + f"{stem}.csv")

        # ── Boxplot : variation totale du CAPEX par secteur ───────────────────────
        capex_long = (
            capex_sector[row_order]
            .stack()
            .reset_index()
        )
        capex_long.columns = ['Sample', 'Sector', 'C_inv_bEUR']
        capex_long['Sector'] = capex_long['Sector'].astype(str)
        row_order_str = [str(s) for s in row_order]

        fig_box = go.Figure()
        for sector in row_order_str:
            vals = capex_long.loc[capex_long['Sector'] == sector, 'C_inv_bEUR']
            fig_box.add_trace(go.Box(
                x=vals,
                name=str(sector),
                orientation='h',
                boxpoints='outliers',
                marker_color='steelblue',
            ))

        fig_box.update_layout(
            title=dict(
                text=f"<b>Variation du CAPEX par secteur{yr_label}</b><br>[b€<sub>2015</sub>]",
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=20),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=13),
            showlegend=False,
            margin=dict(l=180, r=60, t=90, b=60),
            width=700,
            height=max(350, 45 * n_rows),
            yaxis=dict(autorange='reversed'),
            xaxis=dict(title='b€'),
        )

        if plot:
            pio.show(fig_box)

        stem_box = f"cinv_variation_sector{yr_str}"
        fig_box.write_html(out_dir + f"_Raw/{stem_box}.html")
        fig_box.write_image(out_dir + f"{stem_box}.pdf",
                            width=700, height=max(350, 45 * n_rows))

        return corr_matrix

    def graph_corr_capex_opex_commissioning(self, ampl_uq_collector=None,
                                             corr_method='pearson',
                                             construction_prefix='cp_',
                                             plot=True):
        """
        Heatmap 2 lignes : corrélation entre CAPEX total (ligne 1), OPEX total (ligne 2)
        et les temps de commissionnement cp_* (colonnes).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        for key in ('C_tot_capex', 'C_tot_opex', 'Samples'):
            if key not in ampl_uq_collector:
                raise ValueError(f"'{key}' introuvable dans le collecteur UQ.")

        # ── CAPEX & OPEX par sample ───────────────────────────────────────────
        capex = ampl_uq_collector['C_tot_capex'].copy().reset_index()
        capex_col = 'C_tot_capex' if 'C_tot_capex' in capex.columns else capex.columns[-1]
        capex = capex[['Sample', capex_col]].copy()
        capex['CAPEX'] = pd.to_numeric(capex[capex_col], errors='coerce') / 1000.0
        capex = capex[['Sample', 'CAPEX']].set_index('Sample')

        opex = ampl_uq_collector['C_tot_opex'].copy().reset_index()
        opex_col = 'C_tot_opex' if 'C_tot_opex' in opex.columns else opex.columns[-1]
        opex = opex[['Sample', opex_col]].copy()
        opex['OPEX'] = pd.to_numeric(opex[opex_col], errors='coerce') / 1000.0
        opex = opex[['Sample', 'OPEX']].set_index('Sample')

        # ── Commissioning times ───────────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        ct_cols = [c for c in samples.columns
                   if isinstance(c, str) and c.startswith(construction_prefix)]
        if not ct_cols:
            raise ValueError(f"Aucune colonne {construction_prefix}* trouvée dans Samples.")

        ct_cols = [c for c in ct_cols if c in ('cp_NUCLEAR', 'cp_WIND_ONSHORE')]
        cp_df = samples[['Sample'] + ct_cols].copy()
        cp_df[ct_cols] = np.ceil(np.exp(cp_df[ct_cols].apply(pd.to_numeric, errors='coerce')))
        cp_df = cp_df.set_index('Sample')

        # ── Fusion & corrélation ──────────────────────────────────────────────
        merged = capex.join(opex, how='inner').join(cp_df, how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides.")

        corr_full = merged.corr(method=corr_method).fillna(0)
        row_matrix = corr_full.loc[['CAPEX', 'OPEX'], ct_cols].copy()

        # Trier les colonnes par |corrélation CAPEX| décroissante
        order = row_matrix.loc['CAPEX'].abs().sort_values(ascending=False).index
        row_matrix = row_matrix[order]

        ct_labels = [self.uncert_param_meaning.get(c, c) for c in row_matrix.columns]

        if not plot:
            return row_matrix

        out_dir = self.outdir + "CapexOpexCommissioning/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)
        row_matrix.to_csv(out_dir + "capex_opex_commissioning.csv")

        self._export_corr_heatmap(
            row_matrix=row_matrix,
            x_labels=ct_labels,
            y_labels=['Total CAPEX', 'Total OPEX'],
            title='CAPEX & OPEX × commissioning time correlation',
            zmin=-1, zmax=1,
            out_dir=out_dir,
            filename_stem='capex_opex_commissioning',
            decimals=2,
            text_threshold=0.4,
            interpolation='nearest',
        )
        return row_matrix

    def graph_corr_co2_commissioning(self, ampl_uq_collector=None,
                                      year_start=None, year_end=None,
                                      normalized=True, corr_method='pearson',
                                      construction_prefix='cp_',
                                      construction_cols=None,
                                      plot=True):
        """
        Corrélation entre le CO2 total capturé (somme de Year_balance['CO2_CAPTURED']
        sur toutes les années) et les temps de commissionnement (cp_*).
        Même format heatmap 1-ligne que graph_corr_capex_commissioning.

        Parameters
        ----------
        year_start / year_end : int | None
            Filtre sur les années. Si None, toutes les années.
        normalized : bool
            True → corrélation ; False → covariance brute.
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        # ── CO2 capturé total par sample ──────────────────────────────────────────
        yb = ampl_uq_collector['Year_balance'].copy()
        if 'CO2_CAPTURED' not in yb.columns:
            raise ValueError("Colonne 'CO2_CAPTURED' introuvable dans Year_balance.")

        yb = yb[['CO2_CAPTURED', 'Sample']].copy().reset_index()
        yb['Years'] = yb['Years'].astype(str).str.replace('YEAR_', '', regex=False)
        yb['CO2_CAPTURED'] = pd.to_numeric(yb['CO2_CAPTURED'], errors='coerce').fillna(0)

        if year_start is not None:
            yb = yb[yb['Years'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            yb = yb[yb['Years'].apply(_year_key) <= int(year_end)]

        # Somme des valeurs positives (capture) par sample
        co2_total = (
            yb[yb['CO2_CAPTURED'] > 0]
            .groupby('Sample')['CO2_CAPTURED']
            .sum() / 1000.0  # GWh → TWh ~ MtCO2
        )
        co2_total.name = 'CO2_captured_total'

        # ── Temps de commissionnement ─────────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)

        if construction_cols is None:
            construction_cols = [c for c in samples.columns
                                  if isinstance(c, str) and c.startswith(construction_prefix)]
            if not construction_cols:
                construction_cols = [c for c in samples.columns
                                      if isinstance(c, str)
                                      and re.search(r'constr|construction|t_constr', c, re.IGNORECASE)]

        construction_cols = [c for c in construction_cols if c in samples.columns]
        if not construction_cols:
            raise ValueError("Aucune colonne de temps de commissionnement détectée dans Samples.")

        cp_df = samples[['Sample'] + construction_cols].copy()
        cp_df[construction_cols] = np.ceil(np.exp(
            cp_df[construction_cols].apply(pd.to_numeric, errors='coerce')
        ))
        cp_df = cp_df.set_index('Sample')

        # ── Fusion & corrélation ──────────────────────────────────────────────────
        merged = co2_total.to_frame().join(cp_df, how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides pour calculer la corrélation.")

        cp_cols = [c for c in construction_cols if c in merged.columns]
        data = merged[['CO2_captured_total'] + cp_cols].apply(pd.to_numeric, errors='coerce')

        if normalized:
            matrix = data.corr(method=corr_method).fillna(0)
            zmin, zmax = -1, 1
            suffix = 'normalized'
        else:
            matrix = data.cov().fillna(0)
            max_abs = float(np.nanmax(np.abs(matrix.values))) or 1.0
            zmin, zmax = -max_abs, max_abs
            suffix = 'raw'

        row_values = matrix.loc['CO2_captured_total', cp_cols]
        row_matrix  = pd.DataFrame([row_values.values],
                                    index=['CO2_captured_total'], columns=cp_cols)
        row_matrix  = row_matrix.sort_values(by='CO2_captured_total', axis=1)

        x_labels = [self.uncert_param_meaning.get(c, c) for c in row_matrix.columns]
        y_labels  = ['Total CO2 captured']

        fig = go.Figure(data=go.Heatmap(
            z=row_matrix.values,
            text=np.round(row_matrix.values, 3),
            texttemplate='%{text}',
            textfont=dict(color='black', size=10),
            zsmooth='best',
            x=x_labels,
            y=y_labels,
            zmin=zmin, zmax=zmax,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x}<br>Corr: %{z:.4f}<extra></extra>',
        ))
        fig.update_layout(
            title=dict(text='Total CO2 captured × commissioning time correlation',
                       x=0.5, xanchor='center',
                       font=dict(family='Raleway', size=28)),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=160, r=80, t=90, b=120),
            width=max(750, 60 * len(x_labels)),
            height=280,
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Covariance_CO2Construction/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        fig.write_html(out_dir + f"_Raw/co2_construction_{suffix}.html")
        self._export_corr_heatmap(
            row_matrix, x_labels, y_labels,
            title='Total CO2 captured × commissioning time correlation',
            zmin=zmin, zmax=zmax,
            out_dir=out_dir,
            filename_stem=f"co2_construction_{suffix}",
        )
        row_matrix.to_csv(out_dir + f"co2_construction_{suffix}.csv")

        return row_matrix

    def graph_corr_inv_cost_commissioning(self, ampl_uq_collector=None, year=2030,
                                          year_start=None, year_end=None,
                                          threshold=0.6, corr_method='pearson',
                                          construction_prefix='cp_',
                                          construction_cols=None,
                                          filter_techs=None,
                                          plot=True):
        """
        Matrice de corrélation entre les coûts d'investissement par technologie
        (C_inv_phase_tech) et les temps de commissionnement (cp_*).

        Utiliser year_start/year_end pour une plage (CAPEX sommé sur les phases).
        Utiliser year seul pour une phase unique (rétrocompatible).

        Parameters
        ----------
        filter_techs : list | None
            Si fourni, seules ces technologies apparaissent en lignes
            (remplace la liste d'exclusion par défaut).
            Ex : ['GRID','DHN','H2_ELECTROLYSIS','SMR','ATM_CCS','INDUSTRY_CCS']
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_inv_phase_tech' not in ampl_uq_collector:
            raise ValueError("'C_inv_phase_tech' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        # ── Résolution de la plage d'années ───────────────────────────────────────
        if year_start is not None or year_end is not None:
            ys = int(year_start) if year_start is not None else 0
            ye = int(year_end)   if year_end   is not None else 9999
            label = f"{ys}–{ye}"
        else:
            ys = int(year)
            ye = int(year)
            label = str(year)

        # ── Coûts d'investissement par technologie pour la plage d'années ─────────
        inv = ampl_uq_collector['C_inv_phase_tech'].copy().reset_index()

        phase_col = 'Phases' if 'Phases' in inv.columns else inv.columns[0]
        tech_col  = 'Technologies' if 'Technologies' in inv.columns else inv.columns[1]

        inv['_year_start'] = inv[phase_col].astype(str).str.split('_').str[0].astype(int)
        if ys == ye:
            inv = inv[inv['_year_start'] == ys].copy()
        else:
            inv = inv[(inv['_year_start'] >= ys) & (inv['_year_start'] <= ye)].copy()

        if inv.empty:
            raise ValueError(f"Aucune donnée C_inv_phase_tech pour la plage {label}.")
        if 'C_inv_phase_tech' not in inv.columns:
            raise ValueError("Colonne 'C_inv_phase_tech' introuvable.")
        if 'Sample' not in inv.columns:
            raise ValueError("Colonne 'Sample' introuvable dans C_inv_phase_tech.")
        year_str = label

        # Pivot : Sample × Technologies
        inv_pivot = (
            inv.groupby(['Sample', tech_col])['C_inv_phase_tech']
            .sum()
            .unstack(fill_value=0)
        )
        # Supprimer les technologies avec variance nulle (coût identique dans tous les scénarios)
        inv_pivot = inv_pivot.loc[:, inv_pivot.std(ddof=0) > 1e-9]

        if filter_techs is not None:
            # Garder uniquement les technologies demandées
            inv_pivot = inv_pivot[[c for c in filter_techs if c in inv_pivot.columns]]
        else:
            # Filtre par défaut : exclure les technos électriques dont le coût
            # est piloté directement par leur propre cp_* (évite auto-corrélation)
            _exclude_techs = ['WIND_ONSHORE', 'NUCLEAR', 'NUCLEAR_SMR', 'PV_FIELD', 'GEOTHERMAL']
            inv_pivot = inv_pivot[[c for c in inv_pivot.columns if c not in _exclude_techs]]

        if inv_pivot.empty:
            raise ValueError("Toutes les technologies ont une variance nulle pour cette phase.")

        # ── Temps de commissionnement ─────────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)

        if construction_cols is None:
            construction_cols = [c for c in samples.columns
                                  if isinstance(c, str) and c.startswith(construction_prefix)]
            if not construction_cols:
                construction_cols = [c for c in samples.columns
                                      if isinstance(c, str)
                                      and re.search(r'constr|construction|t_constr', c, re.IGNORECASE)]

        construction_cols = [c for c in construction_cols if c in samples.columns]
        if not construction_cols:
            raise ValueError("Aucune colonne de temps de commissionnement détectée dans Samples.")

        cp_df = samples[['Sample'] + construction_cols].copy()
        cp_df[construction_cols] = np.ceil(np.exp(
            cp_df[construction_cols].apply(pd.to_numeric, errors='coerce')
        ))
        cp_df = cp_df.set_index('Sample')

        # ── Fusion ────────────────────────────────────────────────────────────────
        merged = inv_pivot.join(cp_df, how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides pour calculer la corrélation.")

        tech_cols = list(inv_pivot.columns)
        cp_cols   = [c for c in construction_cols if c in merged.columns]

        # ── Corrélation : technologies (lignes) × cp_params (colonnes) ────────────
        corr_matrix = (
            merged[tech_cols + cp_cols]
            .corr(method=corr_method)
            .loc[tech_cols, cp_cols]
        )

        # ── Filtrage par threshold ────────────────────────────────────────────────
        keep_rows = corr_matrix.abs().ge(threshold).any(axis=1)
        keep_cols = corr_matrix.abs().ge(threshold).any(axis=0)
        corr_filtered = corr_matrix.loc[keep_rows, keep_cols]

        if corr_filtered.empty:
            print(f"Aucune corrélation ne dépasse le seuil {threshold}. "
                  "La matrice complète est retournée.")
            corr_filtered = corr_matrix

        x_labels = [self.uncert_param_meaning.get(c, c) for c in corr_filtered.columns]
        y_labels = list(corr_filtered.index)

        z_vals = corr_filtered.values.copy().astype(float)
        z_vals[np.abs(z_vals) < threshold] = np.nan
        text_mask = np.where(np.isnan(z_vals), '', np.round(corr_filtered.values, 2).astype(str))

        n_rows = len(y_labels)
        n_cols = len(x_labels)
        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=10),
            x=x_labels,
            y=y_labels,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x}<br>%{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=f"Investment cost × commissioning time ({year_str})",
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=200, r=80, t=90, b=200),
            width=max(600, 70 * n_cols),
            height=max(400, 40 * n_rows),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + f"InvCost_Corr_Commission_{year_str}/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        stem = f"InvCost_corr_commission_{year_str}"
        fig.write_html(out_dir + f"_Raw/{stem}.html")
        z_vals_df = pd.DataFrame(z_vals, index=corr_filtered.index, columns=corr_filtered.columns)
        self._export_corr_heatmap(
            z_vals_df, x_labels, y_labels,
            title=f"Investment cost × commissioning time ({year_str})",
            zmin=-1, zmax=1,
            out_dir=out_dir,
            filename_stem=stem,
            interpolation='nearest',
            text_threshold=threshold,
            bad_color='white',
        )
        corr_filtered.to_csv(out_dir + f"{stem}.csv")

        return corr_filtered

    def graph_inv_cost_variation_techs(self, ampl_uq_collector=None,
                                       technologies=None,
                                       year_start=None, year_end=None,
                                       plot=True):
        """
        Boxplot de la variation inter-scénarios du CAPEX total (somme sur toutes les phases
        ou la plage year_start–year_end) pour une liste de technologies spécifiques.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_inv_phase_tech' not in ampl_uq_collector:
            raise ValueError("'C_inv_phase_tech' introuvable dans le collecteur UQ.")

        _default_techs = [
            'BIOMASS_TO_HVC', 'BIO_HYDROLYSIS', 'DEC_HP_ELEC', 'DEC_THHP_GAS',
            'DHN_BOILER_OIL', 'GRID', 'INDUSTRY_CCS', 'IND_BOILER_COAL',
            'IND_BOILER_GAS', 'IND_BOILER_WOOD', 'METHANOL_TO_HVC', 'PV_RESIDENTIAL',
        ]
        tech_list = technologies if technologies is not None else _default_techs

        inv = ampl_uq_collector['C_inv_phase_tech'].copy().reset_index()
        inv['C_inv_phase_tech'] = pd.to_numeric(inv['C_inv_phase_tech'], errors='coerce').fillna(0)

        phase_col = 'Phases' if 'Phases' in inv.columns else inv.columns[0]
        tech_col  = 'Technologies' if 'Technologies' in inv.columns else inv.columns[1]

        inv['_ys'] = inv[phase_col].astype(str).str.split('_').str[0].astype(int)
        if year_start is not None:
            inv = inv[inv['_ys'] >= int(year_start)]
        if year_end is not None:
            inv = inv[inv['_ys'] <= int(year_end)]

        inv = inv[inv[tech_col].isin(tech_list)]

        if inv.empty:
            print("graph_inv_cost_variation_techs: aucune donnée pour ces technologies.")
            return pd.DataFrame()

        df_agg = inv.groupby(['Sample', tech_col], as_index=False)['C_inv_phase_tech'].sum()
        df_agg.rename(columns={tech_col: 'Technology'}, inplace=True)
        df_agg['C_inv_bEUR'] = df_agg['C_inv_phase_tech'] / 1000.0

        order = (df_agg.groupby('Technology')['C_inv_bEUR']
                 .median().sort_values(ascending=False).index.tolist())

        if not plot:
            return df_agg

        traces = []
        for tech in order:
            vals = df_agg.loc[df_agg['Technology'] == tech, 'C_inv_bEUR']
            label = self.dict_meaning().get(tech, tech.replace('_', ' ').capitalize())
            traces.append(go.Box(
                y=vals, name=label,
                boxpoints='outliers', notched=False,
            ))

        range_label = ''
        if year_start or year_end:
            range_label = f" ({year_start or ''}–{year_end or ''})"

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f'Investment cost variation by technology{range_label} [b€]',
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-45)
        pio.show(fig)

        outdir = self.outdir + "CapexOpex/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/inv_cost_variation_techs_raw.html")

        y_max = round(float(df_agg['C_inv_bEUR'].max()), 2)
        title_str = f"<b>Investment cost variation by technology{range_label}</b><br>[b€<sub>2015</sub>]"
        fig.update_layout(
            title=dict(text=title_str, x=0.5, xanchor='center',
                       font=dict(family='Raleway', size=22)),
            yaxis=dict(range=[0, y_max], tickvals=[0, y_max],
                       ticktext=['0', str(y_max)]),
            showlegend=False,
            template='simple_white',
        )
        fig.update_xaxes(tickangle=-45)
        fig.write_image(outdir + "inv_cost_variation_techs.pdf", width=1400, height=550)
        plt.close()

        return df_agg

    def graph_inv_cost_variation_all(self, ampl_uq_collector=None,
                                     year_start=None, year_end=None,
                                     min_spread_beur=0.1, plot=True):
        """
        Boxplot du CAPEX total par technologie, en gardant uniquement celles dont
        l'écart (max - min) entre scénarios dépasse min_spread_beur [b€].
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_inv_phase_tech' not in ampl_uq_collector:
            raise ValueError("'C_inv_phase_tech' introuvable dans le collecteur UQ.")

        inv = ampl_uq_collector['C_inv_phase_tech'].copy().reset_index()
        inv['C_inv_phase_tech'] = pd.to_numeric(inv['C_inv_phase_tech'], errors='coerce').fillna(0)

        phase_col = 'Phases' if 'Phases' in inv.columns else inv.columns[0]
        tech_col  = 'Technologies' if 'Technologies' in inv.columns else inv.columns[1]

        inv['_ys'] = inv[phase_col].astype(str).str.split('_').str[0].astype(int)
        if year_start is not None:
            inv = inv[inv['_ys'] >= int(year_start)]
        if year_end is not None:
            inv = inv[inv['_ys'] <= int(year_end)]

        df_agg = inv.groupby(['Sample', tech_col], as_index=False)['C_inv_phase_tech'].sum()
        df_agg.rename(columns={tech_col: 'Technology'}, inplace=True)
        df_agg['C_inv_bEUR'] = df_agg['C_inv_phase_tech'] / 1000.0

        # Garder uniquement les techs avec spread > min_spread_beur
        spread = df_agg.groupby('Technology')['C_inv_bEUR'].apply(lambda x: x.max() - x.min())
        keep = spread[spread > min_spread_beur].index
        df_agg = df_agg[df_agg['Technology'].isin(keep)]

        if df_agg.empty:
            print("graph_inv_cost_variation_all: aucune technologie avec variation suffisante.")
            return pd.DataFrame()

        order = (df_agg.groupby('Technology')['C_inv_bEUR']
                 .median().sort_values(ascending=False).index.tolist())

        if not plot:
            return df_agg

        traces = []
        for tech in order:
            vals = df_agg.loc[df_agg['Technology'] == tech, 'C_inv_bEUR']
            label = self.dict_meaning().get(tech, tech.replace('_', ' ').capitalize())
            traces.append(go.Box(
                y=vals, name=label,
                boxpoints='outliers', notched=False,
            ))

        range_label = ''
        if year_start or year_end:
            range_label = f" ({year_start or ''}–{year_end or ''})"

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f'Investment cost variation (spread > {min_spread_beur} b€){range_label}',
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-45)
        pio.show(fig)

        outdir = self.outdir + "CapexOpex/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/inv_cost_variation_all_raw.html")

        y_max = round(float(df_agg['C_inv_bEUR'].max()), 2)
        title_str = f"<b>Investment cost variation{range_label}</b><br>[b€<sub>2015</sub>]"
        fig.update_layout(
            title=dict(text=title_str, x=0.5, xanchor='center',
                       font=dict(family='Raleway', size=22)),
            yaxis=dict(range=[0, y_max], tickvals=[0, y_max],
                       ticktext=['0', str(y_max)]),
            showlegend=False,
            template='simple_white',
        )
        fig.update_xaxes(tickangle=-45)
        fig.write_image(outdir + "inv_cost_variation_all.pdf", width=1400, height=550)
        plt.close()

        return df_agg

    def graph_corr_gwp_op_commissioning(self, ampl_uq_collector=None,
                                        year_start=2030, year_end=2039,
                                        threshold=0.5, corr_method='pearson',
                                        construction_prefix='cp_',
                                        construction_cols=None,
                                        plot=True):
        """
        Matrice de corrélation entre GWP_op agrégé par ressource (Gwp_breakdown)
        et les temps de commissionnement (cp_*), sur la période year_start–year_end.

        Parameters
        ----------
        year_start, year_end : int
            Plage d'années à agréger.
        threshold : float
            Seuil : lignes/colonnes sous ce seuil sont masquées en blanc.
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        construction_prefix : str
            Préfixe des colonnes cp_* dans Samples.
        construction_cols : list|None
            Si fourni, remplace la détection automatique.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Gwp_breakdown' not in ampl_uq_collector:
            raise ValueError("'Gwp_breakdown' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        # ── GWP_op filtré par période ─────────────────────────────────────────────
        gwp = ampl_uq_collector['Gwp_breakdown'].copy().reset_index()

        year_col = 'Years'    if 'Years'    in gwp.columns else gwp.columns[0]
        elem_col = 'Elements' if 'Elements' in gwp.columns else gwp.columns[1]

        gwp[year_col] = gwp[year_col].astype(str).str.replace('YEAR_', '', regex=False)

        def _to_int(v):
            nums = re.findall(r'\d+', str(v))
            return int(nums[0]) if nums else None

        gwp['_yr'] = gwp[year_col].apply(_to_int)
        gwp = gwp[(gwp['_yr'] >= year_start) & (gwp['_yr'] <= year_end)].copy()

        if gwp.empty:
            raise ValueError(f"Aucune donnée Gwp_breakdown entre {year_start} et {year_end}.")
        if 'GWP_op' not in gwp.columns:
            raise ValueError("Colonne 'GWP_op' introuvable dans Gwp_breakdown.")
        if 'Sample' not in gwp.columns:
            raise ValueError("Colonne 'Sample' introuvable dans Gwp_breakdown.")

        gwp['GWP_op'] = pd.to_numeric(gwp['GWP_op'], errors='coerce').fillna(0)
        gwp = gwp[gwp['GWP_op'] > 0]

        # Elements contient déjà les noms de ressources (GAS, COAL, WOOD…)
        # pour les lignes GWP_op — pas besoin de regex de regroupement.
        gwp_pivot = (
            gwp.groupby(['Sample', elem_col])['GWP_op']
            .sum()
            .unstack(fill_value=0)
        )
        gwp_pivot = gwp_pivot.loc[:, gwp_pivot.std(ddof=0) > 1e-9]
        if gwp_pivot.empty:
            raise ValueError("Toutes les ressources ont une variance nulle sur cette période.")

        # ── Temps de commissionnement ─────────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)

        if construction_cols is None:
            construction_cols = [c for c in samples.columns
                                  if isinstance(c, str) and c.startswith(construction_prefix)]
            if not construction_cols:
                construction_cols = [c for c in samples.columns
                                      if isinstance(c, str)
                                      and re.search(r'constr|construction|t_constr', c, re.IGNORECASE)]
        construction_cols = [c for c in construction_cols if c in samples.columns]
        if not construction_cols:
            raise ValueError("Aucune colonne de temps de commissionnement détectée dans Samples.")

        cp_df = samples[['Sample'] + construction_cols].copy()
        cp_df[construction_cols] = np.ceil(np.exp(
            cp_df[construction_cols].apply(pd.to_numeric, errors='coerce')
        ))
        cp_df = cp_df.set_index('Sample')

        # ── Fusion et corrélation ─────────────────────────────────────────────────
        merged = gwp_pivot.join(cp_df, how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides pour calculer la corrélation.")

        res_cols = list(gwp_pivot.columns)
        cp_cols  = [c for c in construction_cols if c in merged.columns]

        corr_matrix = (
            merged[res_cols + cp_cols]
            .corr(method=corr_method)
            .loc[res_cols, cp_cols]
        )

        # ── Filtrage par threshold ────────────────────────────────────────────────
        keep_rows = corr_matrix.abs().ge(threshold).any(axis=1)
        keep_cols = corr_matrix.abs().ge(threshold).any(axis=0)
        corr_filtered = corr_matrix.loc[keep_rows, keep_cols]

        if corr_filtered.empty:
            print(f"Aucune corrélation ne dépasse le seuil {threshold}. Matrice complète retournée.")
            corr_filtered = corr_matrix

        x_labels = [self.uncert_param_meaning.get(c, c) for c in corr_filtered.columns]
        y_labels  = list(corr_filtered.index)

        z_vals = corr_filtered.values.copy().astype(float)
        z_vals[np.abs(z_vals) < threshold] = np.nan
        text_mask = np.where(np.isnan(z_vals), '', np.round(corr_filtered.values, 2).astype(str))

        n_rows = len(y_labels)
        n_cols = len(x_labels)
        year_text = f"{year_start}–{year_end}"

        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            text=text_mask,
            texttemplate='%{text}',
            textfont=dict(color='black', size=10),
            x=x_labels,
            y=y_labels,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x}<br>%{y}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=f"GWP_op by resource × commissioning time ({year_text})",
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=14),
            margin=dict(l=150, r=80, t=90, b=200),
            width=max(600, 70 * n_cols),
            height=max(400, 50 * n_rows),
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + f"GWP_op_Corr_Commission_{year_start}_{year_end}/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        stem = f"GWP_op_corr_commission_{year_start}_{year_end}"
        fig.write_html(out_dir + f"_Raw/{stem}.html")
        z_vals_df = pd.DataFrame(z_vals, index=corr_filtered.index, columns=corr_filtered.columns)
        self._export_corr_heatmap(
            z_vals_df, x_labels, y_labels,
            title=f"GWP_op by resource × commissioning time ({year_text})",
            zmin=-1, zmax=1,
            out_dir=out_dir,
            filename_stem=stem,
            interpolation='nearest',
            text_threshold=threshold,
            bad_color='white',
        )
        corr_filtered.to_csv(out_dir + f"{stem}.csv")

        return corr_filtered

    def graph_corr_resource_commissioning(self, ampl_uq_collector=None,
                                           year_start=2020, year_end=2050, year_step=5,
                                           threshold=0.3, corr_method='pearson',
                                           construction_prefix='cp_',
                                           construction_cols=None,
                                           plot=True):
        """
        Produit une heatmap de corrélation par année (multiples de year_step de year_start
        à year_end). Chaque heatmap : ressources en x, commissioning times en y.

        Parameters
        ----------
        year_start / year_end / year_step : int
            Plage d'années (ex. 2020, 2050, 5 → 7 figures).
        threshold : float
            Valeurs sous ce seuil (|r|) masquées en blanc.
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        construction_prefix : str
            Préfixe des colonnes cp_* dans Samples.
        construction_cols : list | None
            Si fourni, utilise ces colonnes plutôt que la détection automatique.
        """
        
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Resources' not in ampl_uq_collector:
            raise ValueError("'Resources' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        # ── Temps de commissionnement (commun à toutes les années) ────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)

        if construction_cols is None:
            construction_cols = [c for c in samples.columns
                                  if isinstance(c, str) and c.startswith(construction_prefix)]
            if not construction_cols:
                construction_cols = [c for c in samples.columns
                                      if isinstance(c, str)
                                      and re.search(r'constr|construction|t_constr', c, re.IGNORECASE)]

        construction_cols = [c for c in construction_cols if c in samples.columns]
        if not construction_cols:
            raise ValueError("Aucune colonne de temps de commissionnement détectée dans Samples.")

        cp_df = samples[['Sample'] + construction_cols].copy()
        cp_df[construction_cols] = np.ceil(np.exp(
            cp_df[construction_cols].apply(pd.to_numeric, errors='coerce')
        ))
        cp_df = cp_df.set_index('Sample')

        # ── Ressources (toutes les années chargées une fois) ──────────────────────
        res_all = ampl_uq_collector['Resources'].copy().reset_index()
        res_all['Years'] = res_all['Years'].astype(str).str.replace('YEAR_', '', regex=False)
        res_all['Res'] = pd.to_numeric(res_all['Res'], errors='coerce').fillna(0) / 1000.0

        out_dir = self.outdir + "Resource_Corr_Commission/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        years = [str(y) for y in range(int(year_start), int(year_end) + 1, int(year_step))]
        all_results = {}

        for year_str in years:
            res_y = res_all[res_all['Years'] == year_str].copy()
            if res_y.empty:
                continue

            # Pivot : Sample × Resources
            res_pivot = (
                res_y.groupby(['Sample', 'Resources'])['Res']
                .sum()
                .unstack(fill_value=0)
            )
            res_pivot = res_pivot.loc[:, res_pivot.std(ddof=0) > 1e-9]
            if res_pivot.empty:
                continue

            merged = res_pivot.join(cp_df, how='inner').dropna()
            if merged.shape[0] < 2:
                continue

            res_cols = list(res_pivot.columns)
            cp_cols  = [c for c in construction_cols if c in merged.columns]

            corr_matrix = (
                merged[res_cols + cp_cols]
                .corr(method=corr_method)
                .loc[cp_cols, res_cols]
            )

            keep_rows = corr_matrix.abs().ge(threshold).any(axis=1)
            keep_cols = corr_matrix.abs().ge(threshold).any(axis=0)
            corr_filtered = corr_matrix.loc[keep_rows, keep_cols]

            if corr_filtered.empty:
                corr_filtered = corr_matrix

            x_labels = list(corr_filtered.columns)
            y_labels  = [self.uncert_param_meaning.get(c, c) for c in corr_filtered.index]

            z_vals = corr_filtered.values.copy().astype(float)
            z_vals[np.abs(z_vals) < threshold] = np.nan
            text_mask = np.where(np.isnan(z_vals), '', np.round(corr_filtered.values, 2).astype(str))

            n_rows = len(y_labels)
            n_cols = len(x_labels)
            fig = go.Figure(data=go.Heatmap(
                z=z_vals,
                text=text_mask,
                texttemplate='%{text}',
                textfont=dict(color='black', size=10),
                x=x_labels,
                y=y_labels,
                zmin=-1, zmax=1,
                colorscale='RdBu_r',
                colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
                hovertemplate='%{x}<br>%{y}<br>Corr: %{z:.3f}<extra></extra>',
            ))
            fig.update_layout(
                title=dict(
                    text=f"Resource consumption × commissioning time ({year_str})",
                    x=0.5, xanchor='center',
                    font=dict(family='Raleway', size=28),
                ),
                template='simple_white',
                font=dict(color='rgb(90,90,90)', size=14),
                margin=dict(l=200, r=80, t=90, b=150),
                width=max(600, 70 * n_cols),
                height=max(400, 40 * n_rows),
                xaxis=dict(tickangle=-45),
                yaxis=dict(autorange='reversed'),
            )

            if plot:
                pio.show(fig)

            stem = f"resource_corr_commission_{year_str}"
            fig.write_html(out_dir + f"_Raw/{stem}.html")
            z_vals_df = pd.DataFrame(z_vals, index=corr_filtered.index, columns=corr_filtered.columns)
            self._export_corr_heatmap(
                z_vals_df, x_labels, y_labels,
                title=f"Resource consumption × commissioning time ({year_str})",
                zmin=-1, zmax=1,
                out_dir=out_dir,
                filename_stem=stem,
                interpolation='nearest',
                text_threshold=threshold,
                bad_color='white',
            )
            corr_filtered.to_csv(out_dir + f"{stem}.csv")
            all_results[year_str] = corr_filtered

        return all_results

    def graph_corr_resource_total_commissioning(self, ampl_uq_collector=None,
                                                  threshold=0.6, corr_method='pearson',
                                                  construction_prefix='cp_',
                                                  min_total_twh=1.0, plot=True):
        """
        Heatmap de corrélation entre la consommation totale de chaque ressource
        (somme sur toute la transition) et les commissioning times cp_*.

        Rows = ressources, colonnes = cp_*.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        for key in ('Resources', 'Samples'):
            if key not in ampl_uq_collector:
                raise ValueError(f"'{key}' introuvable dans le collecteur UQ.")

        # ── Commissioning times ───────────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        ct_cols = [c for c in samples.columns
                   if isinstance(c, str) and c.startswith(construction_prefix)
                   and 'CCGT' not in c.upper()]
        if not ct_cols:
            raise ValueError(f"Aucune colonne {construction_prefix}* trouvée dans Samples.")

        cp_df = samples[['Sample'] + ct_cols].copy()
        cp_df[ct_cols] = np.ceil(np.exp(cp_df[ct_cols].apply(pd.to_numeric, errors='coerce')))
        cp_df = cp_df.set_index('Sample')

        # ── Consommation totale par ressource × sample ────────────────────────
        res = ampl_uq_collector['Resources'].copy().reset_index()
        res['Res'] = pd.to_numeric(res['Res'], errors='coerce').fillna(0) / 1000.0

        res_col = next((c for c in res.columns
                        if c not in ('Sample', 'Years', 'Res') and 'Res' not in c), None)
        if res_col is None:
            res_col = [c for c in res.columns if c not in ('Sample', 'Years', 'Res')][0]

        total = (res.groupby(['Sample', res_col], as_index=False)['Res'].sum()
                    .pivot_table(index='Sample', columns=res_col, values='Res', aggfunc='sum')
                    .fillna(0))

        # Retirer ressources non pertinentes
        exclude_res = {'RES_GEO', 'URANIUM', 'RES_WIND'}
        total = total[[c for c in total.columns if c not in exclude_res]]

        # Filtrer ressources insignifiantes
        sig = total.abs().median()
        total = total[sig[sig >= min_total_twh].index]

        if total.empty:
            raise ValueError("Aucune ressource significative après filtrage.")

        # ── Corrélation ───────────────────────────────────────────────────────
        merged = total.join(cp_df, how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides.")

        corr_full  = merged.corr(method=corr_method).fillna(0)
        res_names  = list(total.columns)
        row_matrix = corr_full.loc[res_names, ct_cols].copy()

        # Masquer sous le seuil
        row_matrix_display = row_matrix.copy()
        row_matrix_display[row_matrix_display.abs() < threshold] = 0.0

        # Retirer lignes/colonnes toutes nulles
        row_matrix_display = row_matrix_display.loc[
            row_matrix_display.abs().max(axis=1) >= threshold]
        row_matrix_display = row_matrix_display[
            [c for c in ct_cols if row_matrix_display[c].abs().max() >= threshold]]

        if row_matrix_display.empty:
            print("Aucune corrélation au-dessus du seuil.")
            return row_matrix

        ct_labels  = [self.uncert_param_meaning.get(c, c) for c in row_matrix_display.columns]
        res_labels = list(row_matrix_display.index)

        if not plot:
            return row_matrix

        out_dir = self.outdir + "ResourceTotalCorrelation/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)
        row_matrix.to_csv(out_dir + "resource_total_commissioning.csv")

        self._export_corr_heatmap(
            row_matrix=row_matrix_display,
            x_labels=ct_labels,
            y_labels=res_labels,
            title='Total resource consumption × commissioning time',
            zmin=-1, zmax=1,
            out_dir=out_dir,
            filename_stem='resource_total_commissioning',
            decimals=2,
            text_threshold=threshold,
            interpolation='nearest',
        )
        return row_matrix

    def graph_covariance_decided_capacity_construction_time(self, ampl_uq_collector=None,
                                                            decided_col=None, construction_cols=None,
                                                            construction_prefix='cp_', normalized=True,
                                                            corr_method='pearson', plot=True,
                                                            include_ccgt=False):
        """
        Corrélation/covariance entre capacité décidée et temps de construction.

        Les temps de construction (cp_*) sont exponentiés avant calcul
        (hypothèse log-normale) et comparés à la capacité décidée de la même
        technologie dans Decision_tracking.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")
        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            # samples.csv has no explicit Sample column; row n corresponds to sample n+1.
            samples['Sample'] = np.arange(1, len(samples) + 1)
        samples['Sample'] = pd.to_numeric(samples['Sample'], errors='coerce')

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy().reset_index()

        if decided_col is None:
            if 'F_decided_realized_up_to' in decision_tracking.columns:
                decided_col = 'F_decided_realized_up_to'
            elif 'F_decided_up_to' in decision_tracking.columns:
                decided_col = 'F_decided_up_to'

        if decided_col is None or decided_col not in decision_tracking.columns:
            raise ValueError(
                f"Colonne de capacité décidée introuvable dans Decision_tracking. "
                f"Colonnes disponibles: {list(decision_tracking.columns)}"
            )

        if 'Sample' not in decision_tracking.columns or 'Technologies' not in decision_tracking.columns:
            raise ValueError("Decision_tracking doit contenir les colonnes 'Sample' et 'Technologies'.")

        if construction_cols is None:
            cols_by_prefix = [
                c for c in samples.columns
                if isinstance(c, str) and c.startswith(construction_prefix)
            ]
            if len(cols_by_prefix) > 0:
                construction_cols = cols_by_prefix
            else:
                construction_cols = [
                    c for c in samples.columns
                    if isinstance(c, str) and re.search(r'constr|construction|t_constr', c, flags=re.IGNORECASE)
                ]

        construction_cols = [c for c in construction_cols if c in samples.columns]
        if not include_ccgt:
            construction_cols = [c for c in construction_cols if 'CCGT' not in c.upper()]
        if len(construction_cols) == 0:
            raise ValueError(
                "Aucune colonne de temps de construction détectée dans Samples. "
                "Passez construction_cols explicitement si nécessaire."
            )

        construction_df = samples[['Sample'] + construction_cols].copy()
        for col in construction_cols:
            construction_df[col] = np.exp(pd.to_numeric(construction_df[col], errors='coerce'))

        decision_tracking['Sample'] = pd.to_numeric(decision_tracking['Sample'], errors='coerce')
        decision_tracking[decided_col] = pd.to_numeric(decision_tracking[decided_col], errors='coerce')
        decision_tracking['Technologies'] = decision_tracking['Technologies'].astype(str)

        cap_by_sample_tech = (
            decision_tracking
            .dropna(subset=['Sample', 'Technologies', decided_col])
            .groupby(['Sample', 'Technologies'], as_index=False)[decided_col]
            .sum()
        )

        values = []
        valid_counts = []
        for cp_col in construction_cols:
            tech = str(cp_col)[len(construction_prefix):] if str(cp_col).startswith(construction_prefix) else str(cp_col)
            cap_tech = cap_by_sample_tech.loc[cap_by_sample_tech['Technologies'] == tech, ['Sample', decided_col]].copy()
            if cap_tech.empty:
                values.append(np.nan)
                valid_counts.append(0)
                continue

            merged = construction_df[['Sample', cp_col]].merge(cap_tech, on='Sample', how='inner')
            merged.dropna(subset=[cp_col, decided_col], inplace=True)
            valid_counts.append(int(len(merged)))

            if len(merged) < 2:
                values.append(np.nan)
                continue

            # Si la capacité décidée est quasi-constante (CV < 0.1%), on ignore cette tech
            cap_std  = merged[decided_col].std(ddof=0)
            cap_mean = merged[decided_col].mean()
            if cap_mean == 0 or (cap_std / abs(cap_mean)) < 0.001:
                values.append(np.nan)
                continue

            if normalized:
                corr_val = merged[[decided_col, cp_col]].corr(method=corr_method).iloc[0, 1]
                values.append(float(corr_val))
            else:
                cov_val = float(np.cov(merged[decided_col], merged[cp_col], ddof=1)[0, 1])
                values.append(cov_val)

        row_matrix = pd.DataFrame([values], index=[decided_col], columns=construction_cols)
        # Supprimer les colonnes NaN (capacité quasi-constante)
        row_matrix = row_matrix.loc[:, row_matrix.notna().any(axis=0)]
        row_matrix = row_matrix.sort_values(by=decided_col, axis=1)

        if normalized:
            zmin, zmax = -1, 1
            cb_title = ''
            title = "Sensitivity of newly capacity to commissioning time"
            suffix = 'normalized'
        else:
            finite_vals = np.asarray([v for v in values if pd.notna(v)], dtype=float)
            max_abs = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size > 0 else 1.0
            if max_abs <= 0:
                max_abs = 1.0
            zmin, zmax = -max_abs, max_abs
            cb_title = 'Covariance'
            title = "Sensitivity of newly capacity to commissioning time"
            suffix = 'raw'

        def _wrap_wind(s, sep):
            if 'wind' in str(s).lower():
                words = str(s).split()
                mid = (len(words) + 1) // 2
                return ' '.join(words[:mid]) + sep + ' '.join(words[mid:])
            return s

        x_labels     = [self.uncert_param_meaning.get(c, c) for c in row_matrix.columns]
        x_labels_pdf = [_wrap_wind(self.uncert_param_meaning.get(c, c), '\n') for c in row_matrix.columns]
        y_labels = [f"Commissioned capacity"]

        fig = go.Figure(
            data=go.Heatmap(
                z=row_matrix.values,
                text=np.round(row_matrix.values, 3),
                texttemplate='%{text}',
                textfont=dict(color='black', size=10),
                zsmooth='best',
                x=x_labels,
                y=y_labels,
                zmin=zmin,
                zmax=zmax,
                colorscale='RdBu_r',
                colorbar=dict(title=cb_title, tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
                hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.4f}<extra></extra>'
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center',
                       font=dict(family='Raleway', size=28)),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=120, r=80, t=90, b=120),
            width=750,
            height=280,
            xaxis=dict(tickangle=0),
            yaxis=dict(autorange='reversed')
        )
        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Covariance_DecidedCapacityConstruction/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)

        fig.write_html(out_dir + f"_Raw/Covariance_decided_capacity_construction_{suffix}.html")
        self._export_corr_heatmap(
            row_matrix, x_labels_pdf, y_labels,
            title="Commissionned capacity and commissioning time correlation",
            zmin=zmin, zmax=zmax,
            out_dir=out_dir,
            filename_stem=f"Covariance_decided_capacity_construction_{suffix}",
            xlabel_rotation=0,
            fig_height=1.5,
        )

        matrix_out = row_matrix.copy()
        matrix_out.to_csv(out_dir + f"Covariance_decided_capacity_construction_{suffix}.csv")

        diagnostics = pd.DataFrame({
            'ConstructionColumn': construction_cols,
            'Technology': [str(c)[len(construction_prefix):] if str(c).startswith(construction_prefix) else str(c) for c in construction_cols],
            'DisplayLabel': x_labels,
            'DecidedCapacityColumn': decided_col,
            'N_ValidSamples': valid_counts,
            'Transform': ['exp'] * len(construction_cols)
        })
        diagnostics.to_csv(out_dir + "Covariance_decided_capacity_construction_columns.csv", index=False)

        return matrix_out

    def graph_corr_decided_capacity_matrix(self, ampl_uq_collector=None, plot=True,
                                            year='all', construction_prefix='cp_',
                                            normalized=True, corr_method='pearson',
                                            min_capacity_gw=0.1, include_ccgt=False):
        """
        Matrice complète de corrélation entre la capacité décidée de TOUTES les
        technologies (axe Y) et TOUS les commissioning times cp_* (axe X).

        Contrairement à graph_covariance_decided_capacity_construction_time qui
        ne corrèle chaque tech qu'avec son propre cp_*, cette fonction calcule
        toutes les combinaisons croisées.

        Parameters
        ----------
        year : int | str
            Année cible ou 'all' pour sommer sur toute la transition.
        min_capacity_gw : float
            Seuil médian (GW) sous lequel une technologie est ignorée.
        include_ccgt : bool
            Inclure les colonnes cp_CCGT* (souvent peu significatives).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")
        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        # ── Commissioning times (exp car log-normale) ─────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        samples['Sample'] = pd.to_numeric(samples['Sample'], errors='coerce')

        ct_cols = [c for c in samples.columns
                   if isinstance(c, str) and c.startswith(construction_prefix)]
        if not include_ccgt:
            ct_cols = [c for c in ct_cols if 'CCGT' not in c.upper()]
        if not ct_cols:
            raise ValueError(f"Aucune colonne {construction_prefix}* trouvée dans Samples.")

        df_ct = samples[['Sample'] + ct_cols].copy()
        for col in ct_cols:
            df_ct[col] = np.exp(pd.to_numeric(df_ct[col], errors='coerce'))
        df_ct['Sample'] = df_ct['Sample'].astype(str)

        # ── Capacité décidée par sample × technologie ─────────────────────────
        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()

        decided_col = next((c for c in ('F_decided_realized_up_to', 'F_decided_up_to')
                            if c in dt.columns), None)
        if decided_col is None:
            raise ValueError(f"Colonne de capacité décidée introuvable. "
                             f"Colonnes: {list(dt.columns)}")

        dt['Sample'] = pd.to_numeric(dt['Sample'], errors='coerce').astype(str)
        dt[decided_col] = pd.to_numeric(dt[decided_col], errors='coerce')
        dt['Technologies'] = dt['Technologies'].astype(str)

        if year != 'all':
            phase_col = next((c for c in dt.columns if c in ('Phases', 'Phase')), None)
            if phase_col:
                dt['_phase_start'] = dt[phase_col].astype(str).str.extract(r'(\d{4})').astype(float)
                dt = dt[dt['_phase_start'] == float(year)]

        cap = (dt.dropna(subset=['Sample', 'Technologies', decided_col])
                 .groupby(['Sample', 'Technologies'], as_index=False)[decided_col].sum())

        # Pivot : lignes = samples, colonnes = technologies
        pivot = cap.pivot_table(index='Sample', columns='Technologies',
                                values=decided_col, aggfunc='sum')

        # Garder uniquement les technologies qui ont un cp_* (matrice carrée)
        ct_techs = [c[len(construction_prefix):] for c in ct_cols]
        pivot = pivot[[t for t in ct_techs if t in pivot.columns]]

        # Filtrer techs insignifiantes
        sig = pivot.median()
        pivot = pivot[sig[sig >= min_capacity_gw].index]

        if pivot.empty:
            print("graph_corr_decided_capacity_matrix: aucune technologie significative.")
            return pd.DataFrame()

        # ── Matrice de corrélation ─────────────────────────────────────────────
        df_merged = df_ct.set_index('Sample').join(pivot, how='inner')
        tech_names = list(pivot.columns)

        corr_matrix = pd.DataFrame(index=tech_names, columns=ct_cols, dtype=float)
        for tech in tech_names:
            for ct in ct_cols:
                s = df_merged[[ct, tech]].dropna()
                if len(s) < 3:
                    corr_matrix.loc[tech, ct] = float('nan')
                    continue
                if corr_method == 'spearman':
                    r = s[ct].rank().corr(s[tech].rank())
                else:
                    r = s[ct].corr(s[tech])
                corr_matrix.loc[tech, ct] = round(float(r), 3)

        # Trier par corrélation max absolue décroissante
        corr_matrix['_max'] = corr_matrix[ct_cols].abs().max(axis=1)
        corr_matrix.sort_values('_max', ascending=False, inplace=True)
        corr_matrix.drop(columns='_max', inplace=True)

        meaning = self.dict_meaning()

        def _readable(name):
            raw = str(name)
            if raw in meaning:
                return meaning[raw]
            if raw in self.uncert_param_meaning:
                return self.uncert_param_meaning[raw]
            return raw.replace('_', ' ').capitalize()

        ct_labels   = [_readable(c[len(construction_prefix):] if c.startswith(construction_prefix) else c)
                       for c in ct_cols]
        tech_labels = [_readable(t) for t in corr_matrix.index]

        if not plot:
            return corr_matrix

        year_str = str(year)
        outdir = self.outdir + "DecidedCapacityCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        corr_matrix.to_csv(outdir + f"corr_decided_capacity_ct_{year_str}.csv")

        title = "Correlation between commissioned capacity and commissioning time"
        self._export_corr_heatmap(
            row_matrix=corr_matrix,
            x_labels=ct_labels,
            y_labels=tech_labels,
            title=title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem=f"corr_decided_capacity_ct_{year_str}",
            decimals=2,
            text_threshold=0.1,
            interpolation='nearest',
        )
        return corr_matrix

    def graph_elec_correlation_construction_time(self, ampl_uq_collector=None, year='all',
                                                   construction_cols=None, construction_prefix='cp_',
                                                   corr_method='pearson', plot=True):
        """
        Heatmap 1-ligne : corrélation entre la production d'électricité et chaque
        temps de commissionnement (cp_*), triée par valeur croissante.

        Parameters
        ----------
        year : str|int
            Année analysée ou 'all' pour agréger toutes les années.
        corr_method : str
            'pearson', 'spearman' ou 'kendall'.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        # ── Production d'électricité par sample ───────────────────────────────
        yb = ampl_uq_collector['Year_balance'].copy().reset_index()
        yb['Years'] = yb['Years'].astype(str).str.replace('YEAR_', '', regex=False)

        if str(year).lower() != 'all':
            year_text = str(year)
            yb = yb[yb['Years'].isin({year_text, f'YEAR_{year_text}'})]
            if yb.empty:
                raise ValueError(f"Aucune donnée pour l'année {year_text}.")

        if 'ELECTRICITY' not in yb.columns:
            raise ValueError("Colonne 'ELECTRICITY' introuvable dans Year_balance.")

        elec_prod = (
            yb[yb['ELECTRICITY'] > 0]
            .groupby('Sample', as_index=False)['ELECTRICITY']
            .sum()
            .rename(columns={'ELECTRICITY': 'Elec_TWh'})
        )
        elec_prod['Elec_TWh'] /= 1000.0

        # ── Temps de commissionnement ─────────────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)

        if construction_cols is None:
            construction_cols = [c for c in samples.columns
                                  if isinstance(c, str) and c.startswith(construction_prefix)]
            if not construction_cols:
                construction_cols = [c for c in samples.columns
                                      if isinstance(c, str)
                                      and re.search(r'constr|construction|t_constr', c, re.IGNORECASE)]

        construction_cols = [c for c in construction_cols if c in samples.columns]
        construction_cols = [c for c in construction_cols if 'CCGT' not in str(c).upper()]
        if not construction_cols:
            raise ValueError("Aucune colonne de temps de construction détectée dans Samples.")

        cp_df = samples[['Sample'] + construction_cols].copy()
        for col in construction_cols:
            cp_df[col] = np.ceil(np.exp(pd.to_numeric(cp_df[col], errors='coerce')))

        # ── Print variation totale d'électricité ──────────────────────────────
        print(f"  Electricity production — min:    {elec_prod['Elec_TWh'].min():.1f} TWh")
        print(f"  Electricity production — median: {elec_prod['Elec_TWh'].median():.1f} TWh")
        print(f"  Electricity production — max:    {elec_prod['Elec_TWh'].max():.1f} TWh")

        # ── Fusion et corrélation ─────────────────────────────────────────────
        merged = elec_prod.merge(cp_df, on='Sample', how='inner').dropna()
        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides pour calculer la corrélation.")

        corr_series = (
            merged[['Elec_TWh'] + construction_cols]
            .corr(method=corr_method)
            .loc['Elec_TWh', construction_cols]
            .sort_values()
        )

        row_matrix = pd.DataFrame([corr_series.values],
                                   index=['Elec_TWh'], columns=corr_series.index)

        x_labels     = [self.uncert_param_meaning.get(c, c) for c in row_matrix.columns]
        x_labels_pdf = [
            lbl.replace(' ', '\n', 1) if 'wind' in lbl.lower() else lbl
            for lbl in x_labels
        ]
        y_labels  = ['Electricity production']
        year_str  = 'all' if str(year).lower() == 'all' else str(year)

        fig = go.Figure(data=go.Heatmap(
            z=row_matrix.values,
            text=np.round(row_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(color='black', size=10),
            zsmooth='best',
            x=x_labels,
            y=y_labels,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
            hovertemplate='%{x}<br>Corr: %{z:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text="Electricity production and commissioning time correlation",
                x=0.5, xanchor='center',
                font=dict(family='Raleway', size=28),
            ),
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=120, r=80, t=90, b=120),
            width=750,
            height=280,
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )
        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Electricity/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(out_dir + f"_Raw/Elec_corr_construction_{year_str}.html")
        self._export_corr_heatmap(
            row_matrix, x_labels_pdf, y_labels,
            title="Electricity production and commissioning time correlation",
            zmin=-1, zmax=1,
            out_dir=out_dir,
            filename_stem=f"Elec_corr_construction_{year_str}",
            xlabel_rotation=0,
            fig_height=1.5,
        )

        return corr_series

    def graph_elec_vs_renewable_imports(self, ampl_uq_collector=None,
                                        elec_year_start=2030, elec_year_end=2040,
                                        res_year_start=2030, res_year_end=2050,
                                        resources=None, corr_method='pearson', plot=True):
        """
        Corrélation entre la production électrique totale (elec_year_start–elec_year_end)
        et l'importation totale de ressources renouvelables (res_year_start–res_year_end).

        Par défaut resources = ['H2_RE', 'METHANOL_RE'].
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if resources is None:
            resources = ['H2_RE', 'METHANOL_RE']

        # ── Production électrique totale par sample ───────────────────────────
        yb = ampl_uq_collector['Year_balance'].copy().reset_index()
        year_col = next((c for c in yb.columns if c in ('Years', 'Year')), None)
        yb['_yr'] = yb[year_col].astype(str).str.extract(r'(\d{4})').astype(int)
        yb_elec = yb[(yb['_yr'] >= elec_year_start) & (yb['_yr'] <= elec_year_end)].copy()
        yb_elec['ELECTRICITY'] = pd.to_numeric(yb_elec['ELECTRICITY'], errors='coerce').fillna(0)
        elec_prod = (yb_elec[yb_elec['ELECTRICITY'] > 0]
                     .groupby('Sample')['ELECTRICITY'].sum() / 1000.0)
        elec_prod.name = f'Elec_{elec_year_start}_{elec_year_end}'

        # ── Consommation totale par resource et par sample ────────────────────
        res = ampl_uq_collector['Resources'].copy().reset_index()
        res_year_col = next((c for c in res.columns if c in ('Years', 'Year')), None)
        res['_yr'] = res[res_year_col].astype(str).str.extract(r'(\d{4})').astype(int)
        res_filt = res[(res['_yr'] >= res_year_start) & (res['_yr'] <= res_year_end)]
        res_col = next((c for c in res.columns if c not in ('Sample', 'Years', 'Year', 'Res', '_yr')), 'Resources')

        res_pivots = {}
        for r in resources:
            r_df = res_filt[res_filt[res_col] == r]
            if r_df.empty:
                print(f"  Ressource '{r}' introuvable dans Resources.")
                continue
            r_sum = r_df.groupby('Sample')['Res'].sum() / 1000.0
            r_sum.name = r
            res_pivots[r] = r_sum

        if not res_pivots:
            print("graph_elec_vs_renewable_imports: aucune ressource trouvée.")
            return pd.DataFrame()

        # ── Fusion et corrélation ─────────────────────────────────────────────
        df = pd.DataFrame(elec_prod)
        for name, series in res_pivots.items():
            df = df.join(series, how='inner')
        df.dropna(inplace=True)

        elec_col = elec_prod.name
        results = {}
        for r in res_pivots:
            r_val = df[[elec_col, r]].corr(method=corr_method).iloc[0, 1]
            results[r] = round(float(r_val), 3)
            print(f"  Corr({elec_col}, {r}) = {results[r]:.3f}")

        if not plot:
            return pd.DataFrame(results, index=[elec_col])

        # ── Scatter plots ─────────────────────────────────────────────────────
        outdir = self.outdir + "Electricity/"
        Path(outdir).mkdir(parents=True, exist_ok=True)

        for r in res_pivots:
            fig = px.scatter(
                df, x=r, y=elec_col,
                title=f'Electricity production ({elec_year_start}–{elec_year_end}) vs {r} ({res_year_start}–{res_year_end})<br>r={results[r]:.3f}',
                labels={r: f'{r} [TWh]', elec_col: f'Electricity [TWh]'},
            )
            # Droite de tendance manuelle (sans statsmodels)
            m, b = np.polyfit(df[r], df[elec_col], 1)
            x_line = np.linspace(df[r].min(), df[r].max(), 100)
            fig.add_trace(go.Scatter(x=x_line, y=m * x_line + b,
                                     mode='lines', line=dict(color='red', width=1.5),
                                     showlegend=False))
            pio.show(fig)
            stem = f"elec_vs_{r}_{elec_year_start}_{elec_year_end}"
            fig.write_html(outdir + f"_Raw/{stem}.html")
            fig.write_image(outdir + f"{stem}.pdf", width=600, height=500)
            plt.close()

        return pd.DataFrame(results, index=[elec_col])

    def graph_gas_consumption_vs_nuclear_ct(self, ampl_uq_collector=None, plot=True,
                                            resource='GA', cp_col='cp_NUCLEAR'):
        """
        Spaghetti plot : consommation annuelle de GAS (TWh) par année,
        une ligne par scénario colorée par le commissioning time du nucléaire.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        # ── Consommation de GAS par année et par sample ───────────────────────
        res = ampl_uq_collector['Resources'].copy().reset_index()
        year_col = next((c for c in res.columns if c in ('Years', 'Year')), None)
        res_col  = next((c for c in res.columns if c not in ('Sample', 'Years', 'Year', 'Res')), 'Resources')

        res['_yr'] = res[year_col].astype(str).str.replace('YEAR_', '', regex=False).str.extract(r'(\d{4})')[0]
        gas = res[res[res_col] == resource][['Sample', '_yr', 'Res']].copy()
        gas['Res'] = pd.to_numeric(gas['Res'], errors='coerce').fillna(0) / 1000.0
        gas = gas.groupby(['Sample', '_yr'], as_index=False)['Res'].sum()

        # ── Commissioning time nucléaire par sample ───────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        if cp_col not in samples.columns:
            raise ValueError(f"'{cp_col}' introuvable dans Samples.")

        ct = samples[['Sample', cp_col]].copy()
        ct[cp_col] = np.ceil(np.exp(pd.to_numeric(ct[cp_col], errors='coerce')))

        gas = gas.merge(ct, on='Sample', how='left')
        gas['_yr'] = pd.to_numeric(gas['_yr'], errors='coerce')
        gas.sort_values(['Sample', '_yr'], inplace=True)

        ordered_years = sorted(gas['_yr'].dropna().unique().tolist())

        if not plot:
            return gas

        ct_min = gas[cp_col].min()
        ct_max = gas[cp_col].max()

        # Un seul scénario représentatif par valeur unique de cp (évite les doublons de couleur)
        representative = (gas.groupby([cp_col, 'Sample'])['Res'].mean()
                          .reset_index()
                          .groupby(cp_col)['Sample'].first()
                          .reset_index())
        kept_samples = set(representative['Sample'])

        fig = go.Figure()
        for sample_id, grp in gas[gas['Sample'].isin(kept_samples)].groupby('Sample'):
            ct_val = grp[cp_col].iloc[0]
            norm   = (ct_val - ct_min) / (ct_max - ct_min) if ct_max > ct_min else 0.5
            # bleu (court) → rouge (long)
            r = int(norm * 255)
            b = int((1 - norm) * 255)
            color = f'rgba({r},0,{b},0.85)'
            fig.add_trace(go.Scatter(
                x=grp['_yr'].astype(str), y=grp['Res'],
                mode='lines',
                line=dict(color=color, width=1.2),
                showlegend=False,
                hovertemplate=f'cp={ct_val:.0f}y<br>%{{x}}: %{{y:.1f}} TWh<extra></extra>',
            ))

        ordered_years_str = [str(y) for y in ordered_years]
        first_year = ordered_years_str[0] if ordered_years_str else None

        fig.update_layout(
            title=f'{resource} consumption across scenarios',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years_str),
        )
        pio.show(fig)

        outdir = self.outdir + "Resources/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        stem = f"{resource}_vs_{cp_col}_spaghetti"
        fig.write_html(outdir + f"_Raw/{stem}.html")

        y_min = float(gas['Res'].min())
        y_max = float(gas['Res'].max())
        yvals = [round(y_min, 1), round(y_max, 1)]
        title_str = f"<b>{resource} consumption across scenarios</b><br>[TWh]"
        self.custom_fig(fig, title_str, yvals, xvals=ordered_years_str, type_graph='bar')
        fig.update_xaxes(
            tickvals=ordered_years_str,
            ticktext=[y if (y == first_year or int(y) % 5 == 0) else '' for y in ordered_years_str],
        )
        fig.update_layout(showlegend=False)
        fig.write_image(outdir + f"{stem}.pdf", width=1200, height=550)
        plt.close()

        return gas

    def graph_gas_consumption_nuclear_wind_scenarios(self, ampl_uq_collector=None, plot=True,
                                                     resource='GAS',
                                                     cp_nuclear_col='cp_NUCLEAR',
                                                     cp_wind_col='cp_WIND_ONSHORE',
                                                     year_range=(2020, 2040)):
        """
        3 subplots (Nuclear CT bins: short/medium/long).
        Each subplot: 3 lines (Wind CT bins: short/medium/long).
        For each (nuclear, wind) pair the displayed scenario is the one
        whose *other* CT parameters are closest to their medians, so the
        nuclear x wind effect is isolated.
        Format: same visual style as graph_total_gwp_per_year_scenarios.
        """
        from plotly.subplots import make_subplots

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        # ── GAS consumption per year per sample ──────────────────────────────
        res = ampl_uq_collector['Resources'].copy().reset_index()
        year_col = next((c for c in res.columns if c in ('Years', 'Year')), None)
        res_col  = next((c for c in res.columns if c not in ('Sample', 'Years', 'Year', 'Res')), 'Resources')

        res['_yr'] = (res[year_col].astype(str)
                      .str.replace('YEAR_', '', regex=False)
                      .str.extract(r'(\d{4})')[0])
        gas = res[res[res_col] == resource][['Sample', '_yr', 'Res']].copy()
        gas['Res'] = pd.to_numeric(gas['Res'], errors='coerce').fillna(0) / 1000.0
        gas = gas.groupby(['Sample', '_yr'], as_index=False)['Res'].sum()
        gas['_yr'] = pd.to_numeric(gas['_yr'], errors='coerce')
        gas = gas[(gas['_yr'] >= year_range[0]) & (gas['_yr'] <= year_range[1])]

        # ── Commissioning times (log-space → real years) ─────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        for col in [cp_nuclear_col, cp_wind_col]:
            if col not in samples.columns:
                raise ValueError(f"'{col}' introuvable dans Samples.")

        cp_cols = [c for c in samples.columns if c.startswith('cp_')]
        ct = samples[['Sample'] + cp_cols].copy()
        for col in cp_cols:
            ct[col] = np.ceil(np.exp(pd.to_numeric(ct[col], errors='coerce')))

        # ── Bins (quantile 33/66) ─────────────────────────────────────────────
        def _bins(series):
            q33 = series.quantile(0.33)
            q66 = series.quantile(0.66)
            return [
                (series.min(), q33,               'Short'),
                (q33,          q66,               'Medium'),
                (q66,          series.max() + 1,  'Long'),
            ]

        nuc_bins  = _bins(ct[cp_nuclear_col])
        wind_bins = _bins(ct[cp_wind_col])

        # Other CTs (neither nuclear nor wind) used to find representative sample
        other_cols = [c for c in cp_cols if c != cp_nuclear_col and c != cp_wind_col]
        other_med  = ct[other_cols].median() if other_cols else None
        other_std  = ct[other_cols].std().replace(0, 1) if other_cols else None

        def _representative(nlo, nhi, wlo, whi):
            mask = (
                (ct[cp_nuclear_col] >= nlo) & (ct[cp_nuclear_col] < nhi) &
                (ct[cp_wind_col]    >= wlo) & (ct[cp_wind_col]    < whi)
            )
            sub = ct[mask]
            if sub.empty:
                return None
            if not other_cols:
                return int(sub['Sample'].iloc[0])
            z = ((sub[other_cols] - other_med) / other_std).pow(2).sum(axis=1)
            return int(sub.loc[z.idxmin(), 'Sample'])

        ordered_years     = sorted(gas['_yr'].dropna().unique().tolist())
        ordered_years_str = [str(int(y)) for y in ordered_years]
        first_year = ordered_years_str[0] if ordered_years_str else None

        if not plot:
            return gas

        wind_colors = ['rgb(30,100,220)', 'rgb(130,60,180)', 'rgb(220,40,40)']

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f'Nuclear CT: {b[2]}' for b in nuc_bins],
            shared_yaxes=True,
            horizontal_spacing=0.05,
        )

        all_y = []
        print(f"\nRepresentative samples selected for {resource} consumption ({year_range[0]}–{year_range[1]}):")
        for col_idx, (nlo, nhi, n_label) in enumerate(nuc_bins, start=1):
            for (wlo, whi, w_label), color in zip(wind_bins, wind_colors):
                sid = _representative(nlo, nhi, wlo, whi)
                if sid is None:
                    print(f"  Nuclear {n_label} / Wind {w_label}: no sample found")
                    continue
                ct_nuc  = float(ct.loc[ct['Sample'] == sid, cp_nuclear_col].iloc[0])
                ct_wind = float(ct.loc[ct['Sample'] == sid, cp_wind_col].iloc[0])
                print(f"  Nuclear {n_label} ({ct_nuc:.0f}y) / Wind {w_label} ({ct_wind:.0f}y): sample {sid}")

                sub = gas[gas['Sample'] == sid].sort_values('_yr')
                y_vals = list(sub['Res'])
                all_y.extend(y_vals)

                fig.add_trace(
                    go.Scatter(
                        x=[str(int(r)) for r in sub['_yr']],
                        y=y_vals,
                        mode='lines+markers',
                        name=f'Wind {w_label}',
                        legendgroup=w_label,
                        showlegend=(col_idx == 1),
                        line=dict(color=color, width=2.5),
                        marker=dict(size=6, color=color,
                                    line=dict(color='white', width=1)),
                        hovertemplate=(f'Nuclear {ct_nuc:.0f}y / Wind {ct_wind:.0f}y'
                                       f'<br>%{{x}}: %{{y:.1f}} TWh<extra></extra>'),
                    ),
                    row=1, col=col_idx,
                )

        y_min = min(all_y) if all_y else 0
        y_max = max(all_y) if all_y else 1
        yvals = [round(y_min, 1), round(y_max, 1)]

        tick_text = [y if (y == first_year or int(y) % 5 == 0) else ''
                     for y in ordered_years_str]

        fig.update_xaxes(
            categoryorder='array', categoryarray=ordered_years_str,
            tickvals=ordered_years_str, ticktext=tick_text,
            showgrid=False,
        )
        fig.update_yaxes(
            tickvals=yvals, ticktext=[str(v) for v in yvals],
            range=[y_min - (y_max - y_min) * 0.05,
                   y_max + (y_max - y_min) * 0.05],
            showgrid=True, gridcolor='lightgrey',
        )
        fig.update_layout(
            title=f'{resource} consumption — Nuclear CT × Wind CT representative scenarios',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial', size=12, color='rgb(90,90,90)'),
            legend=dict(
                title='Wind onshore CT',
                x=0.99, y=0.99, xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(90,90,90,0.3)', borderwidth=1,
            ),
        )

        pio.show(fig)

        outdir = self.outdir + "Resources/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        stem = f"{resource}_nuclear_wind_ct_scenarios"
        fig.write_html(outdir + f"_Raw/{stem}.html")
        fig.write_image(outdir + f"{stem}.pdf", width=1800, height=550)
        plt.close()

        return gas

    def graph_correlation_production_construction_time(self, ampl_uq_collector=None, year='all',
                                                       layers=None, construction_cols=None,
                                                       construction_prefix='cp_', normalized=True,
                                                       corr_method='pearson', use_exp=True,
                                                       plot=True):
        """
        Correlation between total sector production and commissioning times.

        X-axis: total production by sector.
        Y-axis: commissioning time parameters.
        Cell value: correlation or covariance across scenarios.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        year_balance = ampl_uq_collector['Year_balance'].copy().reset_index()
        samples = ampl_uq_collector['Samples'].copy()

        year_col = 'Years' if 'Years' in year_balance.columns else year_balance.columns[0]
        element_col = 'Elements' if 'Elements' in year_balance.columns else year_balance.columns[1]
        sample_col = 'Sample' if 'Sample' in year_balance.columns else None
        if sample_col is None:
            year_balance['Sample'] = 0
            sample_col = 'Sample'

        if 'Sample' not in samples.columns:
            samples = samples.copy().reset_index()
            if 'Sample' not in samples.columns:
                samples['Sample'] = samples.index + 1
        else:
            samples = samples.copy().reset_index(drop=True)

        if layers is None:
            layers = ['METHANOL', 'AMMONIA', 'ELECTRICITY', 'GAS', 'H2', 'WOOD', 'WET_BIOMASS', 'HEAT_HIGH_T',
                      'HEAT_LOW_T_DECEN', 'HEAT_LOW_T_DHN', 'HVC',
                      'MOB_FREIGHT_BOAT', 'MOB_FREIGHT_RAIL', 'MOB_FREIGHT_ROAD', 'MOB_PRIVATE',
                      'MOB_PUBLIC']
        if isinstance(layers, str):
            layers = [layers]

        lower_to_col = {str(c).lower(): c for c in year_balance.columns}
        resolved_layers = []
        for layer in layers:
            if layer in year_balance.columns:
                resolved_layers.append(layer)
                continue
            match = lower_to_col.get(str(layer).lower())
            if match is not None:
                resolved_layers.append(match)

        production_layers = [col for col in resolved_layers if col in year_balance.columns]
        if len(production_layers) == 0:
            raise ValueError("Aucune couche de production valide n'est disponible dans Year_balance.")

        if str(year).lower() != 'all':
            year_text = str(year)
            valid_year_keys = {year_text, f'YEAR_{year_text}'}
            year_balance = year_balance.loc[year_balance[year_col].astype(str).isin(valid_year_keys)].copy()
            if year_balance.empty:
                raise ValueError(f"Aucune donnée disponible pour l'année {year_text}.")

        if construction_cols is None:
            cols_by_prefix = [
                c for c in samples.columns
                if isinstance(c, str) and c.startswith(str(construction_prefix))
            ]
            if len(cols_by_prefix) > 0:
                construction_cols = cols_by_prefix
            else:
                construction_cols = [
                    c for c in samples.columns
                    if isinstance(c, str)
                    and re.search(r'constr|construction|t_constr|cp_', c, flags=re.IGNORECASE)
                ]

        construction_cols = [c for c in construction_cols if c in samples.columns and c != 'Sample']
        if len(construction_cols) == 0:
            raise ValueError(
                "Aucune colonne de temps de construction détectée dans Samples. "
                "Passez construction_cols explicitement si nécessaire."
            )

        production_df = year_balance[[sample_col] + production_layers].copy()
        for layer in production_layers:
            production_df[layer] = pd.to_numeric(production_df[layer], errors='coerce').fillna(0.0)
            production_df[layer] = production_df[layer].clip(lower=0.0)

        production_df = production_df.groupby(sample_col, as_index=False)[production_layers].sum()
        production_df[production_layers] = production_df[production_layers] / 1000.0

        construction_df = samples[['Sample'] + construction_cols].copy()
        for col in construction_cols:
            construction_df[col] = pd.to_numeric(construction_df[col], errors='coerce')
        if use_exp:
            construction_df[construction_cols] = np.exp(construction_df[construction_cols])

        merged = production_df.merge(construction_df, left_on=sample_col, right_on='Sample', how='inner')
        merged.dropna(subset=production_layers + construction_cols, how='any', inplace=True)

        if merged.shape[0] < 2:
            raise ValueError("Pas assez de scénarios valides pour calculer la corrélation production/temps de construction.")

        data = merged[production_layers + construction_cols].copy()

        if normalized:
            matrix = data.corr(method=corr_method).fillna(0.0)
            zmin, zmax = -1, 1
            cb_title = ''
            cb_tickvals = [-1, 0, 1]
            cb_ticktext = ['-1', '0', '1']
            suffix = 'normalized'
            title = 'Correlation between total sector production and commissioning time'
        else:
            matrix = data.cov().fillna(0.0)
            max_abs = float(np.nanmax(np.abs(matrix.values))) if matrix.size > 0 else 1.0
            if max_abs <= 0:
                max_abs = 1.0
            zmin, zmax = -max_abs, max_abs
            cb_title = 'Covariance'
            cb_tickvals = None
            cb_ticktext = None
            suffix = 'raw'
            title = 'Covariance between total sector production and commissioning time'

        row_matrix = matrix.loc[construction_cols, production_layers]

        meaning = self.dict_meaning()
        x_labels = [meaning.get(c, c) for c in production_layers]
        y_labels = [self.uncert_param_meaning.get(c, c) for c in construction_cols]

        colorbar_dict = dict(title=cb_title)
        if cb_tickvals is not None:
            colorbar_dict.update(tickvals=cb_tickvals, ticktext=cb_ticktext)

        fig = go.Figure(
            data=go.Heatmap(
                z=row_matrix.values,
                text=np.round(row_matrix.values, 3),
                texttemplate='%{text}',
                textfont=dict(color='black', size=10),
                x=x_labels,
                y=y_labels,
                zmin=zmin,
                zmax=zmax,
                colorscale='RdBu_r',
                colorbar=colorbar_dict,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.4f}<extra></extra>'
            )
        )

        year_label = 'all years' if str(year).lower() == 'all' else str(year)
        fig.update_layout(
            title=f"{title}<br><sup>Year: {year_label}</sup>",
            template='simple_white',
            font=dict(color='rgb(90,90,90)', size=18),
            margin=dict(l=140, r=80, t=100, b=140),
            width=1100,
            height=500
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Covariance_ProductionConstruction/"
        out_dir_raw = out_dir + "_Raw/"
        if not os.path.exists(Path(out_dir)):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(out_dir_raw)):
            Path(out_dir_raw).mkdir(parents=True, exist_ok=True)

        year_suffix = str(year).replace('/', '_')
        fig.write_html(out_dir_raw + f"Correlation_production_construction_{year_suffix}_{suffix}.html")

        matrix_out = row_matrix.copy()
        matrix_out.index = construction_cols
        matrix_out.columns = production_layers

        year_text_label = 'all years' if str(year).lower() == 'all' else str(year)
        self._export_corr_heatmap(
            matrix_out, x_labels, y_labels,
            title=f"Production and commissioning time correlation ({year_text_label})",
            zmin=zmin, zmax=zmax,
            out_dir=out_dir,
            filename_stem=f"Correlation_production_construction_{year_suffix}_{suffix}",
            decimals=1
        )
        matrix_out.to_csv(out_dir + f"Correlation_production_construction_{year_suffix}_{suffix}.csv")

        diagnostics = pd.DataFrame({
            'ProductionLayer': production_layers,
            'ProductionLabel': x_labels
        })
        diagnostics.to_csv(out_dir + f"Correlation_production_construction_{year_suffix}_production_columns.csv", index=False)

        diagnostics_construction = pd.DataFrame({
            'ConstructionColumn': construction_cols,
            'ConstructionLabel': y_labels,
            'Transform': ['exp' if use_exp else 'none'] * len(construction_cols)
        })
        diagnostics_construction.to_csv(out_dir + f"Correlation_production_construction_{year_suffix}_construction_columns.csv", index=False)

        return matrix_out


    def graph_covariance_year_balance(self, ampl_uq_collector=None, year='2050', layers=None,
                                      normalized=True, plot=True, corr_method='pearson',
                                      min_nonzero_share=0.05, min_std=1e-6, zero_tol=1e-9):
        """
        Matrices de covariance/corrélation basées sur Year_balance,
        séparées en production et consommation pour chaque secteur.

        Parameters
        ----------
        year : str|int
            Année analysée (ex: '2050') ou 'all' pour agréger toutes les années.
        layers : list[str] | None
            Couches Year_balance à analyser. Si None, utilise les colonnes
            standards de graph_layer (hors 'Sample').
        normalized : bool
            True => matrice de corrélation ; False => covariance brute.
        corr_method : str
            Méthode de corrélation ('pearson', 'spearman', 'kendall').
        min_nonzero_share : float
            Part minimale de scénarios non-nuls pour conserver un élément.
        min_std : float
            Écart-type minimal pour conserver un élément.
        zero_tol : float
            Tolérance pour considérer une valeur comme nulle.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        year_balance = ampl_uq_collector['Year_balance'].copy().reset_index()

        year_col = 'Years' if 'Years' in year_balance.columns else year_balance.columns[0]
        element_col = 'Elements' if 'Elements' in year_balance.columns else year_balance.columns[1]
        sample_col = 'Sample' if 'Sample' in year_balance.columns else None

        if sample_col is None:
            year_balance['Sample'] = 0
            sample_col = 'Sample'

        if layers is None:
            layers = ['METHANOL', 'AMMONIA', 'ELECTRICITY', 'GAS', 'H2', 'WOOD', 'WET_BIOMASS', 'HEAT_HIGH_T',
                      'HEAT_LOW_T_DECEN', 'HEAT_LOW_T_DHN', 'HVC',
                      'MOB_FREIGHT_BOAT', 'MOB_FREIGHT_RAIL', 'MOB_FREIGHT_ROAD', 'MOB_PRIVATE',
                      'MOB_PUBLIC']

        if isinstance(layers, str):
            layers = [layers]

        # Resolve requested layers with a case-insensitive fallback.
        resolved_layers = []
        lower_to_col = {str(c).lower(): c for c in year_balance.columns}
        for layer in layers:
            if layer in year_balance.columns:
                resolved_layers.append(layer)
                continue
            match = lower_to_col.get(str(layer).lower())
            if match is not None:
                resolved_layers.append(match)

        available_layers = [col for col in resolved_layers if col in year_balance.columns]
        if len(available_layers) == 0:
            raise ValueError("Aucune couche demandée n'est disponible dans Year_balance.")

        if str(year).lower() != 'all':
            year_text = str(year)
            valid_year_keys = {year_text, f'YEAR_{year_text}'}
            year_balance = year_balance.loc[year_balance[year_col].astype(str).isin(valid_year_keys)].copy()
            if year_balance.empty:
                raise ValueError(f"Aucune donnée disponible pour l'année {year_text}.")

        if not os.path.exists(Path(self.outdir + "Covariance_YearBalance/")):
            Path(self.outdir + "Covariance_YearBalance/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Covariance_YearBalance/_Raw/")):
            Path(self.outdir + "Covariance_YearBalance/_Raw/").mkdir(parents=True, exist_ok=True)

        meaning = self.dict_meaning()
        suffix = 'normalized' if normalized else 'raw'
        year_suffix = str(year).replace('/', '_')

        matrices = {}
        diagnostics_rows = []

        for layer in available_layers:
            temp_layer = year_balance[[sample_col, element_col, layer]].copy()
            temp_layer[layer] = pd.to_numeric(temp_layer[layer], errors='coerce').fillna(0.0)

            for flow in ['production', 'consumption']:
                if flow == 'production':
                    temp_flow = temp_layer.loc[temp_layer[layer] > zero_tol].copy()
                    temp_flow['FlowValue'] = temp_flow[layer]
                else:
                    temp_flow = temp_layer.loc[temp_layer[layer] < -zero_tol].copy()
                    temp_flow['FlowValue'] = -temp_flow[layer]

                if temp_flow.empty:
                    diagnostics_rows.append({
                        'Layer': layer,
                        'Flow': flow,
                        'Status': 'empty_after_flow_filter',
                        'KeptElements': 0
                    })
                    continue

                grouped = temp_flow.groupby([sample_col, element_col], as_index=False)['FlowValue'].sum()
                matrix_input = grouped.pivot(index=sample_col, columns=element_col, values='FlowValue').fillna(0.0)

                if matrix_input.shape[1] < 2:
                    diagnostics_rows.append({
                        'Layer': layer,
                        'Flow': flow,
                        'Status': 'not_enough_elements_before_filter',
                        'KeptElements': int(matrix_input.shape[1])
                    })
                    continue

                nonzero_share = (matrix_input.abs() > zero_tol).mean(axis=0)
                std_dev = matrix_input.std(axis=0, ddof=0)
                keep_mask = (nonzero_share >= min_nonzero_share) & (std_dev >= min_std)

                kept_elements = [el for el in matrix_input.columns if bool(keep_mask.get(el, False))]
                if len(kept_elements) < 2:
                    diagnostics_rows.append({
                        'Layer': layer,
                        'Flow': flow,
                        'Status': 'not_enough_elements_after_filter',
                        'KeptElements': int(len(kept_elements))
                    })

                    diag_el = pd.DataFrame({
                        'Element': list(nonzero_share.index),
                        'NonzeroShare': [float(nonzero_share[e]) for e in nonzero_share.index],
                        'StdDev': [float(std_dev[e]) for e in std_dev.index],
                        'Kept': [bool(keep_mask[e]) for e in nonzero_share.index]
                    })
                    diag_el.to_csv(
                        self.outdir + f"Covariance_YearBalance/{layer}_{flow}_{year_suffix}_diagnostics.csv",
                        index=False
                    )
                    continue

                matrix_input = matrix_input[kept_elements]

                if normalized:
                    matrix = matrix_input.corr(method=corr_method).fillna(0)
                    zmin, zmax = -1, 1
                    cb_title = 'Covariance normalisée [-1,1]'
                    title_main = '<b>Matrice de covariance normalisée (corrélation)</b>'
                else:
                    matrix = matrix_input.cov().fillna(0)
                    max_abs = float(np.nanmax(np.abs(matrix.values))) if matrix.size > 0 else 1.0
                    if max_abs <= 0:
                        max_abs = 1.0
                    zmin, zmax = -max_abs, max_abs
                    cb_title = 'Covariance'
                    title_main = '<b>Matrice de covariance</b>'

                labels = [meaning.get(el, el) for el in kept_elements]

                fig = go.Figure(
                    data=go.Heatmap(
                        z=matrix.values,
                        x=labels,
                        y=labels,
                        zmin=zmin,
                        zmax=zmax,
                        colorscale='RdBu_r',
                        colorbar=dict(title=cb_title),
                        hovertemplate='X: %{x}<br>Y: %{y}<br>Valeur: %{z:.2f}<extra></extra>'
                    )
                )

                year_label = str(year)
                if str(year).lower() == 'all':
                    year_text = 'toutes années'
                else:
                    year_text = year_label

                flow_label = 'Production' if flow == 'production' else 'Consommation'

                fig.update_layout(
                    title=(
                        f"{title_main}<br>{layer} - {flow_label} - année {year_text}"
                        f"<br><sup>Filtre: part non-nulle ≥ {min_nonzero_share:.0%}, écart-type ≥ {min_std:g}</sup>"
                    ),
                    template='simple_white',
                    width=1100,
                    height=900,
                    xaxis=dict(title='Éléments', tickangle=-45),
                    yaxis=dict(title='Éléments', autorange='reversed')
                )

                if plot:
                    pio.show(fig)

                fig.write_html(
                    self.outdir + f"Covariance_YearBalance/_Raw/Covariance_{layer}_{flow}_{year_suffix}_{suffix}.html"
                )

                matrix_out = matrix.copy()
                matrix_out.index = kept_elements
                matrix_out.columns = kept_elements

                flow_label = 'Production' if flow == 'production' else 'Consumption'
                year_text_label = 'all years' if str(year).lower() == 'all' else str(year)
                self._export_corr_heatmap(
                    matrix_out, labels, labels,
                    title=f"{layer} — {flow_label} correlation ({year_text_label})",
                    zmin=zmin, zmax=zmax,
                    out_dir=self.outdir + "Covariance_YearBalance/",
                    filename_stem=f"Covariance_{layer}_{flow}_{year_suffix}_{suffix}"
                )
                matrix_out.to_csv(
                    self.outdir + f"Covariance_YearBalance/Covariance_{layer}_{flow}_{year_suffix}_{suffix}.csv"
                )

                diag_el = pd.DataFrame({
                    'Element': list(nonzero_share.index),
                    'NonzeroShare': [float(nonzero_share[e]) for e in nonzero_share.index],
                    'StdDev': [float(std_dev[e]) for e in std_dev.index],
                    'Kept': [bool(keep_mask[e]) for e in nonzero_share.index]
                })
                diag_el.to_csv(
                    self.outdir + f"Covariance_YearBalance/{layer}_{flow}_{year_suffix}_diagnostics.csv",
                    index=False
                )

                matrices[(layer, flow)] = matrix_out
                diagnostics_rows.append({
                    'Layer': layer,
                    'Flow': flow,
                    'Status': 'ok',
                    'KeptElements': int(len(kept_elements))
                })

        diagnostics_summary = pd.DataFrame(diagnostics_rows)
        diagnostics_summary.to_csv(
            self.outdir + f"Covariance_YearBalance/Covariance_year_balance_summary_{year_suffix}_{suffix}.csv",
            index=False
        )

        return matrices
        
    def graph_tech_cap(self, ampl_uq_collector = None, plot = True, technologies=None, color=None):
        """
        Plot installed capacities (F) by year.

        Parameters
        ----------
        ampl_uq_collector : dict | None
            UQ collector. If None, use self.ampl_uq_collector.
        plot : bool
            If True, display plots and export PDF/HTML.
        technologies : str | list[str] | None
            Optional technology filter (e.g., 'NUCLEAR' or ['NUCLEAR', 'NUCLEAR_SMR']).
            If None, keep the previous behavior and plot by sector.
        color : str | None
            Optional color override for box traces. If None, use existing
            dictionary colors. You can pass 'noir' or 'black' for black.
        """
        
        if ampl_uq_collector == None:
            ampl_uq_collector = self.ampl_uq_collector
            
        results = ampl_uq_collector['Assets'].copy()
        results.reset_index(inplace=True)
        results = results.set_index(['Years','Technologies','Sample'])

        selected_tech = None
        if technologies is not None:
            if isinstance(technologies, str):
                selected_tech = [technologies]
            else:
                selected_tech = list(technologies)
            selected_tech = [str(t) for t in selected_tech]

            available_tech = set(results.index.get_level_values('Technologies').astype(str).unique())
            selected_tech = [t for t in selected_tech if t in available_tech]
            if len(selected_tech) == 0:
                raise ValueError("Aucune technologie demandee n'est disponible dans Assets.")

            dict_tech = {'Selected_technologies': selected_tech}
        else:
            dict_tech = self._group_tech_per_eud()

        df_to_plot_full = pd.DataFrame()
        for sector, tech in dict_tech.items():

            temp = results.loc[results.index.get_level_values('Technologies').isin(tech),'F']
            df_to_plot = pd.DataFrame(index=temp.index,columns=['F'])
            for y in temp.index.get_level_values(0).unique():
                temp_y = temp.loc[temp.index.get_level_values('Years') == y]
                temp_y.dropna(how='all',inplace=True)
                if not temp_y.empty:
                    #temp_y = self._remove_low_values(temp_y,threshold=0.05)
                    temp_y = self._remove_low_values(temp_y,threshold=0.0)
                    df_to_plot.update(temp_y)
            df_to_plot.dropna(how='all',inplace=True)

            if df_to_plot.empty:
                continue
            
            df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
            
            df_to_plot.reset_index(inplace=True)
            df_to_plot['Technologies'] = df_to_plot['Technologies'].astype("str")
            df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '')
            
            if plot:
                meaning = self.dict_meaning()
                if selected_tech is not None:
                    display_tech_names = [meaning.get(t, t) for t in selected_tech]
                    plot_title_sector = ", ".join(display_tech_names)
                else:
                    plot_title_sector = sector[0] if isinstance(sector, tuple) else sector

                # if len(df_to_plot.index.get_level_values(0).unique()) <= 1:
                #     fig = px.bar(df_to_plot, x='Years', y = 'F',color='Technologies',
                #                  title=self.case_study + ' - ' + sector+' - Installed capacity',
                #                  color_discrete_map=self.color_dict_full)
                # else:
                    
                fig = px.box(df_to_plot, x='Years', y = 'F',color='Technologies',
                         title= str(plot_title_sector)+' - Installed capacity',
                         color_discrete_map=self.color_dict_full,notched=False, points = 'outliers')
                if color is not None:
                    user_color = str(color).strip().lower()
                    box_color = 'black' if user_color == 'noir' else str(color)
                    fig.update_traces(marker_color=box_color, line_color=box_color, fillcolor=box_color)
                # fig.for_each_trace(lambda trace: trace.update(fillcolor = trace.line.color))
                # fig.update_traces(mode='none')
                
                fig.update_xaxes(categoryorder='array', categoryarray= sorted(df_to_plot['Years'].unique()))
                        
                if len(df_to_plot.index.get_level_values(0).unique()) >= 1:
                    pio.show(fig)
                    if not os.path.exists(Path(self.outdir+"Tech_Cap/")):
                        Path(self.outdir+"Tech_Cap").mkdir(parents=True,exist_ok=True)
                    if not os.path.exists(Path(self.outdir+"Tech_Cap/_Raw/")):
                        Path(self.outdir+"Tech_Cap/_Raw").mkdir(parents=True,exist_ok=True)
                    
                    sector_name = sector[0] if isinstance(sector, tuple) else sector
                    if selected_tech is not None:
                        sector_name = ", ".join([meaning.get(t, t) for t in selected_tech])
                    # Nettoyer le nom du fichier en enlevant les caractères problématiques
                    safe_sector_name = "".join(c for c in str(sector_name) if c.isalnum() or c in (' ', '_', '-')).strip()
                    
                    fig.write_html(self.outdir+"Tech_Cap/_Raw/"+safe_sector_name+"_raw.html")
                    title = "{} - Installed capacities".format(sector_name)
                    temp = df_to_plot.copy()
                    yvals = [0,round(min(temp['F']),1),round(max(temp['F']),1)]
                    
                    self.custom_fig(fig, title, yvals, xvals=sorted(df_to_plot['Years'].unique()), type_graph='bar', y_unit='[GW]')
                        
                    
                    fig.write_image(self.outdir+"Tech_Cap/"+safe_sector_name+".pdf", width=1200, height=550)
                    fig_export = go.Figure(fig)
                    fig_export.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
                    fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
                    fig_export.write_image(self.outdir+"Tech_Cap/"+safe_sector_name+".png", width=1200, height=550)
                plt.close()
            
            if len(df_to_plot_full) == 0:
                df_to_plot_full = df_to_plot
            else:
                df_to_plot_full = df_to_plot_full.append(df_to_plot)
            
        return df_to_plot_full

    def graph_tech_cap_heat_low_t_by_resource(self, ampl_uq_collector=None, plot=True):
        """
        Comme graph_tech_cap mais pour les technologies Heat_low_T, agrégées par
        ressource (dernier token du nom : DHN_BOILER_GAS → GAS, DEC_HP_ELEC → ELEC…).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        heat_techs = list(self.dict_color('Heat_low_T').keys())

        results = ampl_uq_collector['Assets'].copy()
        results.reset_index(inplace=True)
        results = results.set_index(['Years', 'Technologies', 'Sample'])

        temp = results.loc[results.index.get_level_values('Technologies').isin(heat_techs), 'F']
        df_to_plot = pd.DataFrame(index=temp.index, columns=['F'])
        for y in temp.index.get_level_values(0).unique():
            temp_y = temp.loc[temp.index.get_level_values('Years') == y].dropna(how='all')
            if not temp_y.empty:
                temp_y = self._remove_low_values(temp_y, threshold=0.0)
                df_to_plot.update(temp_y)
        df_to_plot.dropna(how='all', inplace=True)

        if df_to_plot.empty:
            raise ValueError("Aucune donnée disponible pour les technologies Heat_low_T.")

        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '', regex=False)

        # Ressource = dernier token du nom de la technologie
        df_to_plot['Resource'] = df_to_plot['Technologies'].astype(str).map(lambda t: t.split('_')[-1])

        # Exclusions
        df_to_plot = df_to_plot[~df_to_plot['Resource'].isin({'H2', 'HYDROLYSIS', 'WASTE', 'BIOMASS', 'SOLAR'})]

        # Agrégation par (Years, Sample, Resource)
        df_agg = df_to_plot.groupby(['Years', 'Sample', 'Resource'], as_index=False)['F'].sum()

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        ordered_years = sorted(df_agg['Years'].unique(), key=_year_key)
        if not plot:
            return df_agg

        res_colors = self.dict_color('Resources')
        token_color_map = {
            'GAS':   res_colors.get('GAS',         'orange'),
            'ELEC':  res_colors.get('ELECTRICITY',  'deepskyblue'),
            'OIL':   res_colors.get('LFO',          'darkviolet'),
            'GEO':   res_colors.get('RES_GEO',      'firebrick'),
            'WOOD':  res_colors.get('WOOD',         'saddlebrown'),
        }

        fig = px.box(df_agg, x='Years', y='F', color='Resource',
                     title='Heat Low-T — Installed capacity by resource',
                     color_discrete_map=token_color_map,
                     points='outliers', notched=False)
        fig.update_xaxes(
            categoryorder='array', categoryarray=ordered_years,
            tickmode='array', tickvals=ordered_years,
            ticktext=[y if _year_key(y) % 5 == 0 else '' for y in ordered_years],
        )

        pio.show(fig)

        Path(self.outdir + "Tech_Cap/").mkdir(parents=True, exist_ok=True)
        Path(self.outdir + "Tech_Cap/_Raw/").mkdir(parents=True, exist_ok=True)
        fig.write_html(self.outdir + "Tech_Cap/_Raw/Heat_low_T_by_resource_raw.html")

        yvals = [0, round(float(df_agg['F'].max()), 1)]
        self.custom_fig(fig, "<b>Heat Low-T — Installed capacity by resource</b>",
                        yvals, xvals=ordered_years, type_graph='bar', y_unit='[GW]')
        fig.write_image(self.outdir + "Tech_Cap/Heat_low_T_by_resource.pdf", width=1200, height=550)
        plt.close()

        return df_agg


    def graph_tech_cap_heat_high_t_by_resource(self, ampl_uq_collector=None, plot=True):
        """
        Comme graph_tech_cap_heat_low_t_by_resource mais pour Heat_high_T.
        Ressource = dernier token du nom (IND_BOILER_GAS → GAS, IND_DIRECT_ELEC → ELEC…).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        heat_techs = list(self.dict_color('Heat_high_T').keys())

        results = ampl_uq_collector['Assets'].copy()
        results.reset_index(inplace=True)
        results = results.set_index(['Years', 'Technologies', 'Sample'])

        temp = results.loc[results.index.get_level_values('Technologies').isin(heat_techs), 'F']
        df_to_plot = pd.DataFrame(index=temp.index, columns=['F'])
        for y in temp.index.get_level_values(0).unique():
            temp_y = temp.loc[temp.index.get_level_values('Years') == y].dropna(how='all')
            if not temp_y.empty:
                temp_y = self._remove_low_values(temp_y, threshold=0.0)
                df_to_plot.update(temp_y)
        df_to_plot.dropna(how='all', inplace=True)

        if df_to_plot.empty:
            raise ValueError("Aucune donnée disponible pour les technologies Heat_high_T.")

        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '', regex=False)
        df_to_plot['Resource'] = df_to_plot['Technologies'].astype(str).map(lambda t: t.split('_')[-1])

        df_to_plot = df_to_plot[~df_to_plot['Resource'].isin({'ELEC'})]

        df_agg = df_to_plot.groupby(['Years', 'Sample', 'Resource'], as_index=False)['F'].sum()

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        ordered_years = sorted(df_agg['Years'].unique(), key=_year_key)

        if not plot:
            return df_agg

        res_colors = self.dict_color('Resources')
        token_color_map = {
            'GAS':   res_colors.get('GAS',        'orange'),
            'ELEC':  res_colors.get('ELECTRICITY', 'deepskyblue'),
            'COAL':  res_colors.get('COAL',        'black'),
            'WOOD':  'saddlebrown',
            'WASTE': res_colors.get('WASTE',       'olive'),
        }

        fig = px.box(df_agg, x='Years', y='F', color='Resource',
                     title='Heat High-T — Installed capacity by resource',
                     color_discrete_map=token_color_map,
                     points='outliers', notched=False)
        fig.update_xaxes(
            categoryorder='array', categoryarray=ordered_years,
            tickmode='array', tickvals=ordered_years,
            ticktext=[y if _year_key(y) % 5 == 0 else '' for y in ordered_years],
        )

        pio.show(fig)

        Path(self.outdir + "Tech_Cap/").mkdir(parents=True, exist_ok=True)
        Path(self.outdir + "Tech_Cap/_Raw/").mkdir(parents=True, exist_ok=True)
        fig.write_html(self.outdir + "Tech_Cap/_Raw/Heat_high_T_by_resource_raw.html")

        yvals = [0, round(float(df_agg['F'].max()), 1)]
        self.custom_fig(fig, "<b>Heat High-T — Installed capacity by resource</b>",
                        yvals, xvals=ordered_years, type_graph='bar', y_unit='[GW]')
        fig.write_image(self.outdir + "Tech_Cap/Heat_high_T_by_resource.pdf", width=1200, height=550)
        plt.close()

        return df_agg


    def graph_layer_heat_high_t_by_resource(self, ampl_uq_collector=None, plot=True,
                                             year_start=None, year_end=None):
        """
        Comme graph_layer pour HEAT_HIGH_T mais agrégé par ressource
        (dernier token du nom : IND_BOILER_GAS → GAS, IND_DIRECT_ELEC → ELEC…).
        Production uniquement (valeurs positives), en TWh.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        heat_techs = list(self.dict_color('Heat_high_T').keys())

        results = ampl_uq_collector['Year_balance'].copy()
        results = results[['HEAT_HIGH_T', 'Sample']]
        results.reset_index(inplace=True)
        results = results.set_index(['Years', 'Elements', 'Sample'])

        temp = results.loc[results.index.get_level_values('Elements').isin(heat_techs), 'HEAT_HIGH_T']
        df_to_plot = pd.DataFrame(index=temp.index, columns=['HEAT_HIGH_T'])
        for y in temp.index.get_level_values('Years').unique():
            temp_y = temp.loc[temp.index.get_level_values('Years') == y].dropna(how='all')
            if not temp_y.empty:
                temp_y = self._remove_low_values(temp_y, threshold=0.01)
                df_to_plot.update(temp_y)
        df_to_plot.dropna(how='all', inplace=True)

        df_to_plot = df_to_plot.loc[df_to_plot['HEAT_HIGH_T'] > 0]
        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '', regex=False)
        df_to_plot['HEAT_HIGH_T'] = pd.to_numeric(df_to_plot['HEAT_HIGH_T'], errors='coerce').fillna(0) / 1000.0

        # Ressource = dernier token
        df_to_plot['Resource'] = df_to_plot['Elements'].astype(str).map(lambda t: t.split('_')[-1])
        df_to_plot = df_to_plot[~df_to_plot['Resource'].isin({'OIL', 'WASTE'})]
        df_agg = df_to_plot.groupby(['Years', 'Sample', 'Resource'], as_index=False)['HEAT_HIGH_T'].sum()

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        ordered_years = sorted(
            [y for y in df_agg['Years'].unique()
             if (year_start is None or _year_key(y) >= int(year_start))
             and (year_end  is None or _year_key(y) <= int(year_end))],
            key=_year_key
        )
        df_agg = df_agg[df_agg['Years'].isin(ordered_years)]

        if not plot:
            return df_agg

        res_colors = self.dict_color('Resources')
        token_color_map = {
            'GAS':   res_colors.get('GAS',        'orange'),
            'ELEC':  res_colors.get('ELECTRICITY', 'deepskyblue'),
            'COAL':  res_colors.get('COAL',        'black'),
            'WOOD':  'saddlebrown',
            'WASTE': res_colors.get('WASTE',       'olive'),
            'OIL':   'blueviolet',
        }

        fig = px.box(df_agg, x='Years', y='HEAT_HIGH_T', color='Resource',
                     title='Heat High-T — Production by resource',
                     color_discrete_map=token_color_map,
                     points='outliers', notched=False)
        n_years = len(ordered_years)
        tick_labels = ordered_years if n_years <= 15 else [y if _year_key(y) % 5 == 0 else '' for y in ordered_years]
        fig.update_xaxes(
            categoryorder='array', categoryarray=ordered_years,
            tickmode='array', tickvals=ordered_years,
            ticktext=tick_labels,
        )

        pio.show(fig)

        Path(self.outdir + "Layers/").mkdir(parents=True, exist_ok=True)
        Path(self.outdir + "Layers/_Raw/").mkdir(parents=True, exist_ok=True)
        fig.write_html(self.outdir + "Layers/_Raw/Heat_high_T_by_resource_prod_raw.html")

        yvals = [0, round(float(df_agg['HEAT_HIGH_T'].max()), 1)]
        self.custom_fig(fig, "<b>Heat High-T — Production by resource</b>",
                        yvals, xvals=ordered_years, type_graph='bar', y_unit='[TWh]')
        fig.write_image(self.outdir + "Layers/Heat_high_T_by_resource_prod.pdf", width=1200, height=550)
        plt.close()

        return df_agg


    def graph_layer_heat_low_t_by_resource(self, ampl_uq_collector=None, plot=True,
                                            year_start=None, year_end=None):
        """
        Comme graph_layer_heat_high_t_by_resource mais pour HEAT_LOW_T
        (somme HEAT_LOW_T_DHN + HEAT_LOW_T_DECEN), agrégée par ressource. Production en TWh.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        heat_techs = list(self.dict_color('Heat_low_T').keys())

        results = ampl_uq_collector['Year_balance'].copy()
        lt_cols = [c for c in results.columns if 'HEAT_LOW_T_DHN' in str(c)] + \
                  [c for c in results.columns if 'HEAT_LOW_T_DECEN' in str(c)]
        if not lt_cols:
            raise ValueError("Aucune colonne HEAT_LOW_T_DHN / HEAT_LOW_T_DECEN dans Year_balance.")
        results = results[lt_cols + ['Sample']]
        results.reset_index(inplace=True)
        results = results.set_index(['Years', 'Elements', 'Sample'])
        results['HEAT_LOW_T'] = results[lt_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

        temp = results.loc[results.index.get_level_values('Elements').isin(heat_techs), 'HEAT_LOW_T']
        df_to_plot = pd.DataFrame(index=temp.index, columns=['HEAT_LOW_T'])
        for y in temp.index.get_level_values('Years').unique():
            temp_y = temp.loc[temp.index.get_level_values('Years') == y].dropna(how='all')
            if not temp_y.empty:
                temp_y = self._remove_low_values(temp_y, threshold=0.01)
                df_to_plot.update(temp_y)
        df_to_plot.dropna(how='all', inplace=True)

        df_to_plot = df_to_plot.loc[df_to_plot['HEAT_LOW_T'] > 0]
        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '', regex=False)
        df_to_plot['HEAT_LOW_T'] = pd.to_numeric(df_to_plot['HEAT_LOW_T'], errors='coerce').fillna(0) / 1000.0

        _tech_to_resource = {
            'DHN_HP_ELEC':     'ELEC',
            'DEC_HP_ELEC':     'ELEC',
            'DEC_DIRECT_ELEC': 'ELEC',
            'DHN_COGEN_GAS':   'GAS',
            'DEC_THHP_GAS':    'GAS',
            'DEC_BOILER_GAS':  'GAS',
            'DEC_BOILER_OIL':  'OIL',
            'DHN_BOILER_OIL':  'OIL',
            'DHN_SOLAR':       'SOLAR',
            'DEC_SOLAR':       'SOLAR',
            'DEC_BOILER_WOOD': 'WOOD',
        }
        df_to_plot['Resource'] = df_to_plot['Elements'].astype(str).map(_tech_to_resource)
        df_to_plot = df_to_plot.dropna(subset=['Resource'])
        df_agg = df_to_plot.groupby(['Years', 'Sample', 'Resource'], as_index=False)['HEAT_LOW_T'].sum()

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        ordered_years = sorted(
            [y for y in df_agg['Years'].unique()
             if (year_start is None or _year_key(y) >= int(year_start))
             and (year_end  is None or _year_key(y) <= int(year_end))],
            key=_year_key
        )
        df_agg = df_agg[df_agg['Years'].isin(ordered_years)]

        if not plot:
            return df_agg

        res_colors = self.dict_color('Resources')
        token_color_map = {
            'GAS':       res_colors.get('GAS',        'orange'),
            'ELEC':      res_colors.get('ELECTRICITY', 'deepskyblue'),
            'OIL':       'blueviolet',
            'GEO':       res_colors.get('RES_GEO',    'firebrick'),
            'WOOD':      'saddlebrown',
            'WASTE':     res_colors.get('WASTE',       'olive'),
            'H2':        res_colors.get('H2',          'violet'),
            'BIOMASS':   'seagreen',
            'SOLAR':     'gold',
            'HYDROLYSIS':'springgreen',
        }

        fig = px.box(df_agg, x='Years', y='HEAT_LOW_T', color='Resource',
                     title='Heat Low-T — Production by resource',
                     color_discrete_map=token_color_map,
                     points='outliers', notched=False)
        n_years = len(ordered_years)
        tick_labels = ordered_years if n_years <= 15 else [y if _year_key(y) % 5 == 0 else '' for y in ordered_years]
        fig.update_xaxes(
            categoryorder='array', categoryarray=ordered_years,
            tickmode='array', tickvals=ordered_years,
            ticktext=tick_labels,
        )

        pio.show(fig)

        Path(self.outdir + "Layers/").mkdir(parents=True, exist_ok=True)
        Path(self.outdir + "Layers/_Raw/").mkdir(parents=True, exist_ok=True)
        fig.write_html(self.outdir + "Layers/_Raw/Heat_low_T_by_resource_prod_raw.html")

        yvals = [0, round(float(df_agg['HEAT_LOW_T'].max()), 1)]
        self.custom_fig(fig, "<b>Heat Low-T — Production by resource</b>",
                        yvals, xvals=ordered_years, type_graph='bar', y_unit='[TWh]')
        fig.write_image(self.outdir + "Layers/Heat_low_T_by_resource_prod.pdf", width=1200, height=550)
        plt.close()

        return df_agg


    def graph_layer_lfo_consumption(self, ampl_uq_collector=None, plot=True,
                                     year_start=None, year_end=None):
        """
        Box plots de la consommation de LFO par technologie (valeurs négatives de
        Year_balance['LFO']), en TWh, par année.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'LFO' not in ampl_uq_collector['Year_balance'].columns:
            raise ValueError("Colonne 'LFO' introuvable dans Year_balance.")

        results = ampl_uq_collector['Year_balance'][['LFO', 'Sample']].copy()
        results.reset_index(inplace=True)
        results = results.set_index(['Years', 'Elements', 'Sample'])

        # Valeurs négatives = consommation
        temp = results.loc[results['LFO'] < 0, 'LFO']
        df_to_plot = pd.DataFrame(index=temp.index, columns=['LFO'])
        for y in temp.index.get_level_values('Years').unique():
            temp_y = temp.loc[temp.index.get_level_values('Years') == y].dropna(how='all')
            if not temp_y.empty:
                temp_y = self._remove_low_values(temp_y, threshold=0.01)
                df_to_plot.update(temp_y)
        df_to_plot.dropna(how='all', inplace=True)
        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '', regex=False)
        df_to_plot['LFO'] = pd.to_numeric(df_to_plot['LFO'], errors='coerce').fillna(0).abs() / 1000.0

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        ordered_years = sorted(
            [y for y in df_to_plot['Years'].unique()
             if (year_start is None or _year_key(y) >= int(year_start))
             and (year_end   is None or _year_key(y) <= int(year_end))],
            key=_year_key
        )
        df_to_plot = df_to_plot[df_to_plot['Years'].isin(ordered_years)]

        if not plot:
            return df_to_plot

        n_years = len(ordered_years)
        tick_labels = ordered_years if n_years <= 15 else [
            y if _year_key(y) % 5 == 0 else '' for y in ordered_years
        ]

        fig = px.box(
            df_to_plot, x='Years', y='LFO', color='Elements',
            title='LFO consumption by technology [TWh]',
            color_discrete_map=self.color_dict_full,
            points='outliers', notched=False,
            category_orders={'Years': ordered_years},
        )
        fig.update_xaxes(
            categoryorder='array', categoryarray=ordered_years,
            tickmode='array', tickvals=ordered_years, ticktext=tick_labels,
        )
        pio.show(fig)

        Path(self.outdir + "Layers/").mkdir(parents=True, exist_ok=True)
        Path(self.outdir + "Layers/_Raw/").mkdir(parents=True, exist_ok=True)
        fig.write_html(self.outdir + "Layers/_Raw/LFO_consumption_raw.html")

        yvals = [0, round(float(df_to_plot['LFO'].max()), 1)]
        self.custom_fig(fig, "<b>LFO consumption by technology</b>",
                        yvals, xvals=ordered_years, type_graph='bar', y_unit='[TWh]')
        fig.update_xaxes(tickmode='array', tickvals=ordered_years, ticktext=tick_labels)
        fig.write_image(self.outdir + "Layers/LFO_consumption.pdf", width=1200, height=550)
        plt.close()

        return df_to_plot

    def graph_layer_lfo_full(self, ampl_uq_collector=None, plot=True,
                              year_start=None, year_end=None):
        """
        Affiche le bilan complet du layer LFO : production (valeurs positives)
        ET consommation (valeurs négatives, affichées en positif) par élément.
        Deux figures séparées pour voir d'où vient et où va le LFO.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'LFO' not in ampl_uq_collector['Year_balance'].columns:
            raise ValueError("Colonne 'LFO' introuvable dans Year_balance.")

        results = ampl_uq_collector['Year_balance'][['LFO', 'Sample']].copy()
        results.reset_index(inplace=True)

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        results['Years'] = results['Years'].astype(str).str.replace('YEAR_', '', regex=False)
        results['LFO'] = pd.to_numeric(results['LFO'], errors='coerce').fillna(0)

        ordered_years = sorted(
            [y for y in results['Years'].unique()
             if (year_start is None or _year_key(y) >= int(year_start))
             and (year_end   is None or _year_key(y) <= int(year_end))],
            key=_year_key
        )
        results = results[results['Years'].isin(ordered_years)]

        n_years = len(ordered_years)
        tick_labels = ordered_years if n_years <= 15 else [
            y if _year_key(y) % 5 == 0 else '' for y in ordered_years
        ]

        outdir = self.outdir + "Layers/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)

        for side, mask, sign, label, fname in [
            ('production', results['LFO'] > 0,  1,  'LFO production by element [TWh]',    'LFO_production'),
            ('consumption', results['LFO'] < 0, -1,  'LFO consumption by element [TWh]',   'LFO_consumption_full'),
        ]:
            df_side = results[mask].copy()
            df_side['LFO'] = df_side['LFO'] * sign / 1000.0

            # médiane par (Year, Element) pour filtrer les éléments insignifiants
            med = df_side.groupby(['Years', 'Elements'])['LFO'].median()
            sig_elems = med[med > 0.01].index.get_level_values('Elements').unique()
            df_side = df_side[df_side['Elements'].isin(sig_elems)]

            if df_side.empty:
                print(f"Aucune donnée pour le côté {side} du layer LFO.")
                continue

            fig = px.box(
                df_side, x='Years', y='LFO', color='Elements',
                title=label,
                color_discrete_map=self.color_dict_full,
                points='outliers', notched=False,
                category_orders={'Years': ordered_years},
            )
            fig.update_xaxes(
                categoryorder='array', categoryarray=ordered_years,
                tickmode='array', tickvals=ordered_years, ticktext=tick_labels,
            )
            pio.show(fig)

            fig.write_html(outdir + f"_Raw/{fname}_raw.html")
            yvals = [0, round(float(df_side['LFO'].max()), 1)]
            self.custom_fig(fig, f"<b>{label}</b>",
                            yvals, xvals=ordered_years, type_graph='bar', y_unit='[TWh]')
            fig.update_xaxes(tickmode='array', tickvals=ordered_years, ticktext=tick_labels)
            fig.write_image(outdir + f"{fname}.pdf", width=1200, height=550)
            plt.close()

    def graph_tech_cap_points(self, ampl_uq_collector=None, plot=True, technologies=None, color=None,
                              jitter=0.35, marker_size=7, marker_opacity=0.75, extra_yticks=None):
        """
        Plot installed capacities (F) by year as a point cloud for selected technologies.

        Parameters
        ----------
        ampl_uq_collector : dict | None
            UQ collector. If None, use self.ampl_uq_collector.
        plot : bool
            If True, display plots and export PDF/HTML/PNG.
        technologies : str | list[str]
            Technology filter (e.g., 'NUCLEAR' or ['NUCLEAR', 'NUCLEAR_SMR']).
            At least one technology must be provided.
        color : str | None
            Optional color override for all points.
            You can pass 'noir' or 'black' for black.
        jitter : float
            Horizontal jitter used to spread points at a given year.
        marker_size : int | float
            Marker size for points.
        marker_opacity : float
            Marker opacity for points.
        extra_yticks : list[float] | None
            Additional tick values to include on the y-axis (e.g. [2, 6]).
            Merged with the automatic min/max ticks.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if technologies is None:
            raise ValueError("Veuillez fournir au moins une technologie via le parametre technologies.")

        results = ampl_uq_collector['Assets'].copy()
        results.reset_index(inplace=True)
        results = results.set_index(['Years', 'Technologies', 'Sample'])

        if isinstance(technologies, str):
            selected_tech = [technologies]
        else:
            selected_tech = list(technologies)
        selected_tech = [str(t) for t in selected_tech]

        available_tech = set(results.index.get_level_values('Technologies').astype(str).unique())
        selected_tech = [t for t in selected_tech if t in available_tech]
        if len(selected_tech) == 0:
            raise ValueError("Aucune technologie demandee n'est disponible dans Assets.")

        temp = results.loc[results.index.get_level_values('Technologies').isin(selected_tech), 'F']
        df_to_plot = pd.DataFrame(index=temp.index, columns=['F'])
        for y in temp.index.get_level_values(0).unique():
            temp_y = temp.loc[temp.index.get_level_values('Years') == y]
            temp_y.dropna(how='all', inplace=True)
            if not temp_y.empty:
                temp_y = self._remove_low_values(temp_y, threshold=0.0)
                df_to_plot.update(temp_y)
        df_to_plot.dropna(how='all', inplace=True)

        if df_to_plot.empty:
            raise ValueError("Aucune donnee disponible pour les technologies demandees.")

        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)

        df_to_plot.reset_index(inplace=True)
        df_to_plot['Technologies'] = df_to_plot['Technologies'].astype("str")
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '')

        if plot:
            meaning = self.dict_meaning()
            display_tech_names = [meaning.get(t, t) for t in selected_tech]
            plot_title = ", ".join(display_tech_names)

            fig = px.strip(
                df_to_plot,
                x='Years',
                y='F',
                color='Technologies',
                title=str(plot_title) + ' - Variation of installed capacity',
                stripmode='overlay',
                color_discrete_map=self.color_dict_full,
                hover_data={'Sample': True}
            )

            fig.update_traces(
                jitter=float(jitter),
                pointpos=0,
                marker=dict(size=float(marker_size), opacity=float(marker_opacity))
            )

            if color is not None:
                user_color = str(color).strip().lower()
                point_color = 'black' if user_color == 'noir' else str(color)
                fig.update_traces(
                    marker_color=point_color,
                    marker_line_color=point_color,
                    marker=dict(size=float(marker_size), opacity=float(marker_opacity), color=point_color)
                )

            fig.update_xaxes(categoryorder='array', categoryarray=sorted(df_to_plot['Years'].unique()))

            pio.show(fig)

            if not os.path.exists(Path(self.outdir + "Tech_Cap_Points/")):
                Path(self.outdir + "Tech_Cap_Points").mkdir(parents=True, exist_ok=True)
            if not os.path.exists(Path(self.outdir + "Tech_Cap_Points/_Raw/")):
                Path(self.outdir + "Tech_Cap_Points/_Raw").mkdir(parents=True, exist_ok=True)

            export_name = ", ".join(display_tech_names)
            safe_export_name = "".join(c for c in str(export_name) if c.isalnum() or c in (' ', '_', '-')).strip()

            fig.write_html(self.outdir + "Tech_Cap_Points/_Raw/" + safe_export_name + "_raw.html")

            title = "{} - Variation of installed capacity across scenarios".format(export_name)
            temp_plot = df_to_plot.copy()
            is_nuclear = any('NUCLEAR' in t.upper() for t in selected_tech)
            if is_nuclear:
                yvals = [0, 2, 4, 6, 8]
            else:
                yvals = [0, round(min(temp_plot['F']), 1), round(max(temp_plot['F']), 1)]
                if extra_yticks is not None:
                    yvals = sorted(set(yvals) | {round(float(v), 1) for v in extra_yticks})
            all_years = sorted(df_to_plot['Years'].unique())
            self.custom_fig(fig, title, yvals, xvals=all_years, type_graph='bar', y_unit='[GW]')
            def _y_sort_key(v):
                nums = re.findall(r'\d+', str(v))
                return int(nums[0]) if nums else 10**9
            first_y = all_years[0] if all_years else None
            fig.update_xaxes(
                tickvals=all_years,
                ticktext=[y if (y == first_y or _y_sort_key(y) % 5 == 0) else '' for y in all_years]
            )

            fig.write_image(self.outdir + "Tech_Cap_Points/" + safe_export_name + ".pdf", width=1200, height=550)

            fig_export = go.Figure(fig)
            fig_export.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
            fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
            fig_export.write_image(self.outdir + "Tech_Cap_Points/" + safe_export_name + ".png", width=1200, height=550)
            plt.close()

        return df_to_plot

    def graph_decided_capacity_bubble(self, ampl_uq_collector=None, plot=True, technologies=None,
                                      color=None, gw_bin=0.1, marker_opacity=0.6,
                                      show_legend_export=False, marker_size_scale=15):
        """
        Plot decided realized capacity (F_decided_realized_up_to) as a fan chart.

        X-axis: Years
        Y-axis: Decided capacity (GW)
        Shaded bands: percentile ranges across scenarios

        Notes
        -----
        The parameters `gw_bin`, `marker_opacity`, and `marker_size_scale` are kept
        for backward compatibility with the previous bubble-chart version.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in decision_tracking.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        year_col = 'Years' if 'Years' in decision_tracking.columns else decision_tracking.columns[0]
        sample_col = 'Sample' if 'Sample' in decision_tracking.columns else None
        tech_col = 'Technologies' if 'Technologies' in decision_tracking.columns else None

        if sample_col is None:
            decision_tracking['Sample'] = 0
            sample_col = 'Sample'

        df = decision_tracking[[sample_col, year_col, 'F_decided_realized_up_to']].copy()
        if technologies is not None:
            if isinstance(technologies, str):
                selected_tech = [technologies]
            else:
                selected_tech = list(technologies)
            selected_tech = [str(t) for t in selected_tech]

            if tech_col and tech_col in decision_tracking.columns:
                df_tech = decision_tracking[decision_tracking[tech_col].astype(str).isin(selected_tech)].copy()
                df = df_tech[[sample_col, year_col, 'F_decided_realized_up_to']].copy()

        if df.empty:
            raise ValueError("Aucune donnée disponible après filtrage.")

        df[year_col] = (
            df[year_col]
            .astype(str)
            .str.replace('YEAR_', '', regex=False)
            .str.replace('_', '-', regex=False)
        )
        df['F_decided_realized_up_to'] = pd.to_numeric(df['F_decided_realized_up_to'], errors='coerce').fillna(0.0)
        df = df.dropna(subset=[year_col])

        if df.empty:
            raise ValueError("Aucune donnée de capacité décidée disponible.")

        # One value per sample and year, then remove 0 GW scenarios as requested previously.
        df = (
            df.groupby([sample_col, year_col], as_index=False)['F_decided_realized_up_to']
              .sum()
        )
        df = df.loc[df['F_decided_realized_up_to'] > 0.0].copy()

        if df.empty:
            raise ValueError("Aucune donnée de capacité décidée > 0 GW disponible.")

        def _year_sort_key(value):
            nums = re.findall(r'\d+', str(value))
            return tuple(int(n) for n in nums) if nums else (str(value),)

        fan_data = (
            df.groupby(year_col)['F_decided_realized_up_to']
              .agg(
                  Scenario_Count='count',
                  mean='mean',
                  p10=lambda s: float(np.nanpercentile(s, 10)),
                  p25=lambda s: float(np.nanpercentile(s, 25)),
                  p50=lambda s: float(np.nanpercentile(s, 50)),
                  p75=lambda s: float(np.nanpercentile(s, 75)),
                  p90=lambda s: float(np.nanpercentile(s, 90))
              )
              .reset_index()
        )
        fan_data['Years'] = fan_data[year_col].astype(str)
        fan_data['__order'] = fan_data['Years'].map(_year_sort_key)
        fan_data = fan_data.sort_values('__order').drop(columns='__order').reset_index(drop=True)

        if fan_data.empty:
            raise ValueError("Aucune statistique de fan chart n'a pu être générée.")

        export_name = "Decided_Capacity_FanChart"
        tech_label = "All Technologies"
        if technologies is not None:
            if isinstance(technologies, str):
                tech_label = self.dict_meaning().get(technologies, technologies)
                safe_tech = "".join(c for c in str(technologies) if c.isalnum() or c in (' ', '_', '-')).strip()
            else:
                tech_list = [self.dict_meaning().get(str(t), str(t)) for t in technologies]
                tech_label = ", ".join(tech_list)
                safe_tech = "_".join("".join(c for c in str(t) if c.isalnum() or c in (' ', '_', '-')).strip() for t in technologies)
            export_name += "_" + safe_tech

        if color is None:
            line_color = 'deeppink'
        else:
            user_color = str(color).strip().lower()
            if user_color in ('noir', 'black'):
                line_color = 'black'
            elif user_color in ('gris', 'gray', 'grey'):
                line_color = 'gray'
            else:
                line_color = str(color)

        if str(line_color).lower() == 'black':
            outer_fill = 'rgba(0,0,0,0.12)'
            inner_fill = 'rgba(0,0,0,0.24)'
        elif str(line_color).lower() == 'gray':
            outer_fill = 'rgba(128,128,128,0.16)'
            inner_fill = 'rgba(128,128,128,0.30)'
        else:
            outer_fill = 'rgba(255,20,147,0.14)'
            inner_fill = 'rgba(255,20,147,0.28)'

        years_sorted = fan_data['Years'].tolist()
        title_raw = f"{tech_label} - Decided Realized Capacity Fan Chart"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years_sorted,
            y=fan_data['p90'].tolist(),
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='P90'
        ))
        fig.add_trace(go.Scatter(
            x=years_sorted,
            y=fan_data['p10'].tolist(),
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty',
            fillcolor=outer_fill,
            hoverinfo='skip',
            showlegend=bool(show_legend_export),
            name='P10-P90'
        ))
        fig.add_trace(go.Scatter(
            x=years_sorted,
            y=fan_data['p75'].tolist(),
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='P75'
        ))
        fig.add_trace(go.Scatter(
            x=years_sorted,
            y=fan_data['p25'].tolist(),
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty',
            fillcolor=inner_fill,
            hoverinfo='skip',
            showlegend=bool(show_legend_export),
            name='P25-P75'
        ))
        fig.add_trace(go.Scatter(
            x=years_sorted,
            y=fan_data['p50'].tolist(),
            mode='lines+markers',
            line=dict(color=line_color, width=3),
            marker=dict(size=7, color=line_color),
            customdata=np.stack([
                fan_data['Scenario_Count'].to_numpy(dtype=float),
                fan_data['p10'].to_numpy(dtype=float),
                fan_data['p25'].to_numpy(dtype=float),
                fan_data['p75'].to_numpy(dtype=float),
                fan_data['p90'].to_numpy(dtype=float),
                fan_data['mean'].to_numpy(dtype=float)
            ], axis=-1),
            hovertemplate=(
                'Year: %{x}<br>'
                'Scenarios: %{customdata[0]:.0f}<br>'
                'P10-P90: %{customdata[1]:.2f} - %{customdata[4]:.2f} GW<br>'
                'P25-P75: %{customdata[2]:.2f} - %{customdata[3]:.2f} GW<br>'
                'Median: %{y:.2f} GW<br>'
                'Mean: %{customdata[5]:.2f} GW<extra></extra>'
            ),
            showlegend=bool(show_legend_export),
            name='Median'
        ))

        fig.update_layout(title=dict(text=title_raw, x=0.5, xanchor='center'))
        fig.update_xaxes(categoryorder='array', categoryarray=years_sorted)

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Decided_Capacity_Bubble/")):
            Path(self.outdir + "Decided_Capacity_Bubble/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Decided_Capacity_Bubble/_Raw/")):
            Path(self.outdir + "Decided_Capacity_Bubble/_Raw/").mkdir(parents=True, exist_ok=True)

        fig.write_html(self.outdir + f"Decided_Capacity_Bubble/_Raw/{export_name}_raw.html")

        y_max = float(fan_data['p90'].max()) if len(fan_data) > 0 else 0.0
        y_mid = float(fan_data['p50'].median()) if len(fan_data) > 0 else 0.0
        yvals = sorted(set([0.0, round(y_mid, 1), round(y_max, 1)]))
        self.custom_fig(fig, title_raw, yvals, xvals=years_sorted, type_graph='bar', y_unit='[GW]')

        fig_export = go.Figure(fig)
        fig_export.update_layout(
            showlegend=bool(show_legend_export),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.write_image(self.outdir + f"Decided_Capacity_Bubble/{export_name}.pdf", width=1200, height=550)
        fig_export.write_image(self.outdir + f"Decided_Capacity_Bubble/{export_name}.png", width=1200, height=550)
        plt.close()

        csv_out = fan_data[[year_col, 'Scenario_Count', 'mean', 'p10', 'p25', 'p50', 'p75', 'p90']].copy()
        csv_out.to_csv(self.outdir + f"Decided_Capacity_Bubble/{export_name}.csv", index=False)

        return fan_data

    def graph_probability_capacity_target(self, ampl_uq_collector=None, technology='NUCLEAR', target_gw=8.0,
                                          tolerance_gw=0.25, smooth_points=300, plot=True,
                                          show_legend_export=False, color=None, connect_points=False):
        """
        Probability curve per year that installed capacity F is close to a target value.

        The yearly probability is estimated empirically on UQ samples:
        P(|F - target| <= tolerance).

        A smooth continuous curve is then fitted across years.

        Parameters
        ----------
        show_legend_export : bool
            If True, keep legend in saved PDF/PNG. If False, hide legend in saved files.
        color : str | None
            Optional line/marker color override. If None, use technology color from
            dict_color('Electricity'). You can pass 'noir'/'black' for black,
            or 'gris'/'gray'/'grey' for gray.
        connect_points : bool
            If True, draw a smoothed connecting line. If False (default),
            show only unconnected yearly points.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Assets' not in ampl_uq_collector:
            raise ValueError("'Assets' introuvable dans le collecteur UQ.")

        assets = ampl_uq_collector['Assets'].copy().reset_index()
        if 'F' not in assets.columns:
            raise ValueError("Colonne 'F' introuvable dans 'Assets'.")

        year_col = 'Years' if 'Years' in assets.columns else assets.columns[0]
        tech_col = 'Technologies' if 'Technologies' in assets.columns else assets.columns[1]
        sample_col = 'Sample' if 'Sample' in assets.columns else None
        if sample_col is None:
            assets['Sample'] = 0
            sample_col = 'Sample'

        df = assets.loc[assets[tech_col] == technology, [sample_col, year_col, 'F']].copy()
        if df.empty:
            raise ValueError(f"Aucune donnee trouvee pour la technologie {technology}.")

        df['F'] = pd.to_numeric(df['F'], errors='coerce').fillna(0.0)
        df = df.groupby([sample_col, year_col], as_index=False)['F'].sum()

        def _year_to_int(value):
            nums = re.findall(r'\d+', str(value))
            return int(nums[0]) if len(nums) > 0 else None

        df['Year_int'] = df[year_col].map(_year_to_int)
        df = df.loc[df['Year_int'].notna()].copy()
        if df.empty:
            raise ValueError("Impossible d'extraire les annees depuis la colonne Years.")

        df['Year_int'] = df['Year_int'].astype(int)

        prob_df = (
            df.assign(InTarget=(df['F'] - float(target_gw)).abs() <= float(tolerance_gw))
              .groupby('Year_int', as_index=False)['InTarget']
              .mean()
              .rename(columns={'InTarget': 'Probability'})
              .sort_values('Year_int')
        )

        x_year = prob_df['Year_int'].to_numpy(dtype=float)
        y_prob = prob_df['Probability'].to_numpy(dtype=float)

        x_smooth = np.linspace(float(x_year.min()), float(x_year.max()), int(max(60, smooth_points))) if len(x_year) >= 2 else x_year.copy()
        if len(x_year) >= 3:
            try:
                from scipy.interpolate import interp1d
                # No extrapolation outside observed years.
                interp_fun = interp1d(x_year, y_prob, kind='quadratic', bounds_error=False, fill_value=np.nan)
                y_smooth = np.clip(interp_fun(x_smooth), 0.0, 1.0)
                if np.isnan(y_smooth).any():
                    y_smooth = np.interp(x_smooth, x_year, y_prob)
            except Exception:
                y_smooth = np.interp(x_smooth, x_year, y_prob)
        elif len(x_year) >= 2:
            y_smooth = np.interp(x_smooth, x_year, y_prob)
        else:
            y_smooth = y_prob.copy()

        fig = go.Figure()
        if color is None:
            tech_color = self.dict_color('Electricity').get(str(technology), 'deeppink')
        else:
            user_color = str(color).strip().lower()
            tech_color = 'black' if user_color == 'noir' else str(color)
        fig.add_trace(
            go.Scatter(
                x=x_year,
                y=y_prob,
                mode='markers',
                marker=dict(size=8, color=tech_color),
                name='Yearly probability'
            )
        )

        meaning = self.dict_meaning()
        tech_label = meaning.get(technology, technology)
        title = "{} - Probability of {} GW installed".format(tech_label, target_gw)
        xvals = sorted(prob_df['Year_int'].astype(int).unique().tolist())
        yvals = [round(v, 1) for v in np.arange(0.0, 1.0001, 0.2)]
        self.custom_fig(fig, title, yvals, xvals=xvals, type_graph='bar')
        fig.update_xaxes(
            tickmode='array',
            tickvals=xvals,
            ticktext=[str(y) for y in xvals]
        )
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            template='simple_white',
            width=1200,
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Probability/")):
            Path(self.outdir + "Probability/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Probability/_Raw/")):
            Path(self.outdir + "Probability/_Raw/").mkdir(parents=True, exist_ok=True)

        safe_tech = "".join(c for c in str(technology) if c.isalnum() or c in (' ', '_', '-')).strip()
        base_name = f"Probability_F_target_{safe_tech}_{float(target_gw):.2f}GW_tol_{float(tolerance_gw):.2f}GW"
        fig.write_html(self.outdir + f"Probability/_Raw/{base_name}_raw.html")
        fig_export = go.Figure(fig)
        fig_export.update_layout(
            showlegend=bool(show_legend_export),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.write_image(self.outdir + f"Probability/{base_name}.pdf", width=1200, height=550)
        fig_export.write_image(self.outdir + f"Probability/{base_name}.png", width=1200, height=550)

        prob_df_out = prob_df.copy()
        prob_df_out['Technology'] = technology
        prob_df_out['Target_GW'] = float(target_gw)
        prob_df_out['Tolerance_GW'] = float(tolerance_gw)
        prob_df_out.to_csv(self.outdir + f"Probability/{base_name}.csv", index=False)

        return prob_df_out


    def graph_decided_realized_up_to_electricity(self, ampl_uq_collector=None, plot=True):
        """
        Plot F_decided_realized_up_to for electricity technologies only,
        with the same visual style as graph_tech_cap (boxplots).

        Parameters
        ----------
        ampl_uq_collector : dict, optional
            UQ collector dictionary.
        plot : bool
            If True, display figure.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        results = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in results.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col = 'Phases' if 'Phases' in results.columns else results.columns[0]
        tech_col = 'Technologies' if 'Technologies' in results.columns else results.columns[1]
        sample_col = 'Sample' if 'Sample' in results.columns else None
        if sample_col is None:
            results['Sample'] = 0
            sample_col = 'Sample'

        elec_colors = {
            tech: color for tech, color in self.dict_color('Electricity').items()
            if tech != 'ELECTRICITY'
        }
        elec_techs = list(elec_colors.keys())
        results = results.loc[results[tech_col].isin(elec_techs)].copy()
        if results.empty:
            raise ValueError("Aucune donnée Decision_tracking pour les technologies électriques.")

        df = results[[sample_col, phase_col, tech_col, 'F_decided_realized_up_to']].copy()
        df.rename(columns={'F_decided_realized_up_to': 'Decision_GW'}, inplace=True)

        phase_order = sorted(df[phase_col].dropna().unique().tolist(), key=self._phase_sort_key)
        phase_order_map = {phase: i for i, phase in enumerate(phase_order)}

        grouped = (
            df
            .groupby([sample_col, tech_col, phase_col], as_index=False)['Decision_GW']
            .sum()
        )
        grouped['__phase_order'] = grouped[phase_col].map(phase_order_map)
        grouped.sort_values(by=[sample_col, tech_col, '__phase_order'], inplace=True)

        grouped['Decision_GW'] = grouped.groupby([sample_col, tech_col])['Decision_GW'].diff().fillna(grouped['Decision_GW'])
        grouped['Decision_GW'] = grouped['Decision_GW'].clip(lower=0)

        grouped['Technologies'] = grouped[tech_col].astype(str)
        grouped = grouped.loc[grouped['Technologies'].isin(elec_techs)].copy()
        if grouped.empty:
            raise ValueError("Aucune technologie électrique restante après agrégation.")

        grouped['Phase'] = grouped[phase_col].astype(str).str.replace('_', '-', regex=False)
        ordered_phase_labels = [str(phase).replace('_', '-') for phase in phase_order]

        fig = px.box(
            grouped,
            x='Phase',
            y='Decision_GW',
            color='Technologies',
            title='Electricity - decided realized capacity up to phase',
            color_discrete_map=elec_colors,
            category_orders={'Technologies': elec_techs},
            notched=False,
            points='outliers'
        )
        fig.update_xaxes(categoryorder='array', categoryarray=ordered_phase_labels)

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Decision_tracking/")):
            Path(self.outdir + "Decision_tracking/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Decision_tracking/_Raw/")):
            Path(self.outdir + "Decision_tracking/_Raw/").mkdir(parents=True, exist_ok=True)

        mode_label = 'increment'
        fig.write_html(self.outdir + f"Decision_tracking/_Raw/F_decided_realized_up_to_electricity_{mode_label}_raw.html")

        title = "Electricity - Decided realized capacity up to phase"
        y_min = float(grouped['Decision_GW'].min()) if len(grouped) > 0 else 0.0
        y_max = float(grouped['Decision_GW'].max()) if len(grouped) > 0 else 0.0
        yvals = [round(y_min, 1), round(y_max, 1)] if y_min != y_max else [round(y_min, 1)]
        self.custom_fig(fig, title, yvals, xvals=ordered_phase_labels, type_graph='bar', y_unit='[GW]')
        fig.write_image(self.outdir + f"Decision_tracking/F_decided_realized_up_to_electricity_{mode_label}.pdf", width=1200, height=550)
        plt.close()

        csv_out = grouped[[sample_col, tech_col, phase_col, 'Phase', 'Decision_GW']].copy()
        csv_out.to_csv(self.outdir + f"Decision_tracking/F_decided_realized_up_to_electricity_{mode_label}.csv", index=False)

        return csv_out

    def graph_decision_probability_curve(self, technology='NUCLEAR', ampl_uq_collector=None, plot=True,
                                         color=None, show_legend_export=False):
        """
        Cumulative (sorted) curve of decided capacity for a technology.
        
        X-axis represents scenario index (sorted), Y-axis represents capacity in GW.
        
        Parameters
        ----------
        technology : str
            Technology to plot (e.g., 'NUCLEAR').
        ampl_uq_collector : dict, optional
            UQ collector dictionary. If None, use self.ampl_uq_collector.
        plot : bool
            If True, display the figure.
        color : str, optional
            Color override for the marker/line. If None, use technology color.
            You can pass 'noir'/'black' for black, or 'gris'/'gray'/'grey' for gray.
        show_legend_export : bool
            If True, keep legend in saved PDF/PNG. If False, hide legend in saved files.
        """
        
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        # 1. Extract data
        results = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        
        if 'F_decided_realized_up_to' not in results.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable.")

        # Identify columns (Sample, Tech, etc.)
        sample_col = 'Sample' if 'Sample' in results.columns else results.columns[0]
        tech_col = 'Technologies' if 'Technologies' in results.columns else None
        
        # Filter by technology
        if tech_col:
            df_tech = results[results[tech_col] == technology].copy()
        else:
            df_tech = results.copy()

        if df_tech.empty:
            raise ValueError(f"Aucune donnée trouvée pour la technologie : {technology}")

        # 2. Prepare data for sorted plot
        # Sum F_decided_realized_up_to across all years for each scenario, then sort
        data_sorted = df_tech.groupby(sample_col)['F_decided_realized_up_to'].sum().sort_values().reset_index()
        data_sorted['Scenario_Index'] = range(1, len(data_sorted) + 1)
        data_sorted['Capacity_GW'] = data_sorted['F_decided_realized_up_to']

        # 3. Determine color
        if color is None:
            tech_color = self.dict_color('Electricity').get(str(technology), 'deeppink')
        else:
            user_color = str(color).strip().lower()
            tech_color = 'black' if user_color == 'noir' else ('gray' if user_color in ('gris', 'grey') else str(color))

        # 4. Create figure with go.Scatter (consistent with other graph functions)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Scenario_Index'].to_numpy(),
                y=data_sorted['Capacity_GW'].to_numpy(),
                mode='markers',
                marker=dict(size=6, color=tech_color),
                name='Decided Capacity',
                showlegend=True
            )
        )

        # 5. Get labels and prepare axes
        meaning = self.dict_meaning()
        tech_label = meaning.get(technology, technology)
        title_main = "{} - Decided Capacity Distribution".format(tech_label)
        title_with_unit = title_main
        
        y_min = float(data_sorted['Capacity_GW'].min())
        y_max = float(data_sorted['Capacity_GW'].max())
        yvals = [round(y_min, 1), round(y_max, 1)] if y_min != y_max else [round(y_min, 1)]

        xvals = list(range(1, len(data_sorted) + 1, max(1, len(data_sorted) // 10)))
        # Ensure a dedicated x tick where the maximum decided capacity is first reached.
        max_reached_idx = int(data_sorted.loc[data_sorted['Capacity_GW'].idxmax(), 'Scenario_Index'])
        xvals = sorted(set(xvals + [max_reached_idx, int(data_sorted['Scenario_Index'].max())]))

        # 6. Apply custom_fig styling
        self.custom_fig(fig, title_with_unit, yvals, xvals=xvals, type_graph='bar', y_unit='[GW]')
        fig.update_layout(
            title=dict(
                text=title_main,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Number of Scenarios',
            template='simple_white',
            width=1200,
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )

        if plot:
            pio.show(fig)

        # 7. Create output directories
        if not os.path.exists(Path(self.outdir + "Decision_tracking/")):
            Path(self.outdir + "Decision_tracking/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Decision_tracking/_Raw/")):
            Path(self.outdir + "Decision_tracking/_Raw/").mkdir(parents=True, exist_ok=True)

        # 8. Save files
        safe_tech = "".join(c for c in str(technology) if c.isalnum() or c in (' ', '_', '-')).strip()
        base_name = f"Decision_Probability_Curve_{safe_tech}"
        
        fig.write_html(self.outdir + f"Decision_tracking/_Raw/{base_name}_raw.html")
        
        fig_export = go.Figure(fig)
        fig_export.update_layout(
            showlegend=bool(show_legend_export),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.write_image(self.outdir + f"Decision_tracking/{base_name}.pdf", width=1200, height=550)
        fig_export.write_image(self.outdir + f"Decision_tracking/{base_name}.png", width=1200, height=550)

        # 9. Export CSV
        csv_out = data_sorted[['Scenario_Index', 'Capacity_GW']].copy()
        csv_out['Technology'] = technology
        csv_out.to_csv(self.outdir + f"Decision_tracking/{base_name}.csv", index=False)

        return csv_out

    def graph_transition_duration_pdf(self, technology='NUCLEAR', ampl_uq_collector=None, plot=True,
                                      color=None, nbins=25, min_decision_gw=1e-6,
                                      show_legend_export=False):
        """
        Empirical PDF of transition duration for one technology across scenarios.

        For each scenario, the decision increments are computed from
        F_decided_realized_up_to across phases (diff per phase). The transition
        duration is then defined as:
            last_decision_phase_center - first_decision_phase_center

        Parameters
        ----------
        technology : str
            Technology to plot (e.g., 'NUCLEAR').
        ampl_uq_collector : dict | None
            UQ collector dictionary. If None, use self.ampl_uq_collector.
        plot : bool
            If True, display the figure.
        color : str | None
            Color override for histogram. If None, use technology color.
            You can pass 'noir'/'black' for black, or 'gris'/'gray'/'grey' for gray.
        nbins : int
            Number of bins for the histogram-based PDF.
        min_decision_gw : float
            Minimum incremental decision (GW) to consider a phase active.
        show_legend_export : bool
            If True, keep legend in saved PDF/PNG. If False, hide legend in saved files.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in decision_tracking.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable.")

        sample_col = 'Sample' if 'Sample' in decision_tracking.columns else decision_tracking.columns[0]
        phase_col = 'Phases' if 'Phases' in decision_tracking.columns else decision_tracking.columns[1]
        tech_col = 'Technologies' if 'Technologies' in decision_tracking.columns else None

        if tech_col is not None:
            df_tech = decision_tracking.loc[decision_tracking[tech_col] == technology].copy()
        else:
            df_tech = decision_tracking.copy()

        if df_tech.empty:
            raise ValueError(f"Aucune donnee trouvee pour la technologie : {technology}")

        df_tech = df_tech[[sample_col, phase_col, 'F_decided_realized_up_to']].copy()
        df_tech['F_decided_realized_up_to'] = pd.to_numeric(df_tech['F_decided_realized_up_to'], errors='coerce').fillna(0.0)

        grouped = (
            df_tech
            .groupby([sample_col, phase_col], as_index=False)['F_decided_realized_up_to']
            .sum()
            .rename(columns={'F_decided_realized_up_to': 'Decision_GW_cum'})
        )

        phase_order = sorted(grouped[phase_col].dropna().unique().tolist(), key=self._phase_sort_key)
        phase_order_map = {phase: i for i, phase in enumerate(phase_order)}
        grouped['__phase_order'] = grouped[phase_col].map(phase_order_map)
        grouped.sort_values(by=[sample_col, '__phase_order'], inplace=True)

        grouped['Decision_GW'] = grouped.groupby(sample_col)['Decision_GW_cum'].diff().fillna(grouped['Decision_GW_cum'])
        grouped['Decision_GW'] = grouped['Decision_GW'].clip(lower=0)

        grouped['PhaseCenter'] = grouped[phase_col].map(self._phase_center_numeric)
        grouped = grouped.loc[grouped['Decision_GW'] >= float(min_decision_gw)].copy()

        if grouped.empty:
            raise ValueError("Aucune decision incrementale positive detectee avec le seuil donne.")

        duration_df = (
            grouped
            .groupby(sample_col, as_index=False)
            .agg(
                FirstDecisionYear=('PhaseCenter', 'min'),
                LastDecisionYear=('PhaseCenter', 'max'),
                TotalDecisionGW=('Decision_GW', 'sum')
            )
        )
        duration_df['TransitionDurationYears'] = duration_df['LastDecisionYear'] - duration_df['FirstDecisionYear']

        duration_values = duration_df['TransitionDurationYears'].to_numpy(dtype=float)
        if len(duration_values) == 0:
            raise ValueError("Aucune duree de transition exploitable apres filtrage.")

        if color is None:
            tech_color = self.dict_color('Electricity').get(str(technology), 'deeppink')
        else:
            user_color = str(color).strip().lower()
            tech_color = 'black' if user_color == 'noir' else ('gray' if user_color in ('gris', 'grey') else str(color))

        nbins = max(5, int(nbins))

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=duration_values,
                histnorm='probability density',
                nbinsx=nbins,
                marker=dict(color=tech_color, line=dict(color=tech_color, width=1)),
                opacity=0.7,
                name='Empirical PDF'
            )
        )

        meaning = self.dict_meaning()
        tech_label = meaning.get(technology, technology)
        n_scen = int(duration_df[sample_col].nunique())
        title = "{} - PDF of transition duration ({} scenarios)".format(tech_label, n_scen)

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title='Transition duration [years]',
            yaxis_title='Probability density',
            template='simple_white',
            width=1200,
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            bargap=0.04
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Decision_tracking/")):
            Path(self.outdir + "Decision_tracking/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Decision_tracking/_Raw/")):
            Path(self.outdir + "Decision_tracking/_Raw/").mkdir(parents=True, exist_ok=True)

        safe_tech = "".join(c for c in str(technology) if c.isalnum() or c in (' ', '_', '-')).strip()
        base_name = f"Decision_Transition_Duration_PDF_{safe_tech}"

        fig.write_html(self.outdir + f"Decision_tracking/_Raw/{base_name}_raw.html")
        fig_export = go.Figure(fig)
        fig_export.update_layout(
            showlegend=bool(show_legend_export),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.write_image(self.outdir + f"Decision_tracking/{base_name}.pdf", width=1200, height=550)
        fig_export.write_image(self.outdir + f"Decision_tracking/{base_name}.png", width=1200, height=550)

        csv_out = duration_df.copy()
        csv_out['Technology'] = technology
        csv_out.to_csv(self.outdir + f"Decision_tracking/{base_name}.csv", index=False)

        return csv_out

    def graph_decision_share_per_year(self, technology='NUCLEAR', ampl_uq_collector=None, plot=True,
                                      color=None, show_legend_export=False,
                                      min_decision_gw=1e-6):
        """
        Share of decided quantity by year for one technology across all scenarios.

        The function computes incremental decisions from F_decided_realized_up_to,
        aggregates them over all samples per year, then normalizes by the total
        decided quantity so that yearly shares sum to 1.

        Parameters
        ----------
        technology : str
            Technology to analyze (e.g., 'NUCLEAR').
        ampl_uq_collector : dict | None
            UQ collector dictionary. If None, use self.ampl_uq_collector.
        plot : bool
            If True, display the figure.
        color : str | None
            Optional color override.
        show_legend_export : bool
            If True, keep legend in saved PDF/PNG. If False, hide legend.
        min_decision_gw : float
            Minimum incremental decision (GW) kept when aggregating by year.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        results = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in results.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable.")

        sample_col = 'Sample' if 'Sample' in results.columns else results.columns[0]
        phase_col = 'Phases' if 'Phases' in results.columns else results.columns[1]
        tech_col = 'Technologies' if 'Technologies' in results.columns else None

        if tech_col is not None:
            df_tech = results.loc[results[tech_col] == technology].copy()
        else:
            df_tech = results.copy()

        if df_tech.empty:
            raise ValueError(f"Aucune donnee trouvee pour la technologie : {technology}")

        grouped = (
            df_tech
            .assign(F_decided_realized_up_to=pd.to_numeric(df_tech['F_decided_realized_up_to'], errors='coerce').fillna(0.0))
            .groupby([sample_col, phase_col], as_index=False)['F_decided_realized_up_to']
            .sum()
            .rename(columns={'F_decided_realized_up_to': 'Decision_GW_cum'})
        )

        phase_order = sorted(grouped[phase_col].dropna().unique().tolist(), key=self._phase_sort_key)
        phase_order_map = {phase: i for i, phase in enumerate(phase_order)}
        grouped['__phase_order'] = grouped[phase_col].map(phase_order_map)
        grouped.sort_values(by=[sample_col, '__phase_order'], inplace=True)

        grouped['Decision_GW'] = grouped.groupby(sample_col)['Decision_GW_cum'].diff().fillna(grouped['Decision_GW_cum'])
        grouped['Decision_GW'] = grouped['Decision_GW'].clip(lower=0)
        grouped = grouped.loc[grouped['Decision_GW'] >= float(min_decision_gw)].copy()

        grouped['Year'] = grouped[phase_col].map(self._phase_center_numeric)
        grouped['Year'] = pd.to_numeric(grouped['Year'], errors='coerce')
        grouped = grouped.loc[grouped['Year'].notna()].copy()
        grouped['Year'] = grouped['Year'].round().astype(int)

        if grouped.empty:
            raise ValueError("Aucune decision incrementale exploitable apres filtrage des phases.")

        annual = grouped.groupby('Year', as_index=False)['Decision_GW'].sum()
        total_decision = float(annual['Decision_GW'].sum())
        if np.isclose(total_decision, 0.0):
            raise ValueError("La quantite totale decidee est nulle, impossible de calculer un pourcentage.")

        annual['Share'] = annual['Decision_GW'] / total_decision

        # Keep a full yearly axis (e.g. 2020..2050) for readability.
        if hasattr(self, 'x_axis') and self.x_axis is not None and len(self.x_axis) > 0:
            all_years = sorted([int(y) for y in self.x_axis])
        else:
            all_years = list(range(int(annual['Year'].min()), int(annual['Year'].max()) + 1))

        annual = (
            pd.DataFrame({'Year': all_years})
            .merge(annual[['Year', 'Decision_GW', 'Share']], on='Year', how='left')
            .fillna({'Decision_GW': 0.0, 'Share': 0.0})
        )

        if color is None:
            line_color = self.dict_color('Electricity').get(str(technology), 'gray')
        else:
            user_color = str(color).strip().lower()
            line_color = 'black' if user_color == 'noir' else ('gray' if user_color in ('gris', 'grey') else str(color))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=annual['Year'].to_numpy(dtype=int),
                y=annual['Share'].to_numpy(dtype=float),
                mode='lines+markers',
                marker=dict(size=7, color=line_color),
                line=dict(width=1.6, color=line_color),
                name='Share of total decisions'
            )
        )

        meaning = self.dict_meaning()
        tech_label = meaning.get(technology, technology)
        title = "{} - Share of decided quantity by year".format(tech_label)
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title='Year',
            yaxis_title='Share of total decisions',
            yaxis=dict(tickformat='.0%', range=[0, min(1.0, max(0.05, float(annual['Share'].max()) * 1.1))]),
            template='simple_white',
            width=1200,
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        fig.update_xaxes(tickmode='array', tickvals=annual['Year'].to_numpy(dtype=int), tickangle=90)

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Decision_tracking/")):
            Path(self.outdir + "Decision_tracking/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Decision_tracking/_Raw/")):
            Path(self.outdir + "Decision_tracking/_Raw/").mkdir(parents=True, exist_ok=True)

        safe_tech = "".join(c for c in str(technology) if c.isalnum() or c in (' ', '_', '-')).strip()
        base_name = f"Decision_share_per_year_{safe_tech}"

        fig.write_html(self.outdir + f"Decision_tracking/_Raw/{base_name}_raw.html")
        fig_export = go.Figure(fig)
        fig_export.update_layout(
            showlegend=bool(show_legend_export),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.write_image(self.outdir + f"Decision_tracking/{base_name}.pdf", width=1200, height=550)
        fig_export.write_image(self.outdir + f"Decision_tracking/{base_name}.png", width=1200, height=550)

        csv_out = annual.copy()
        csv_out['Technology'] = technology
        csv_out['Total_Decision_GW'] = total_decision
        csv_out.to_csv(self.outdir + f"Decision_tracking/{base_name}.csv", index=False)

        return csv_out

    def graph_commissioning_time_for_decided_capacity(self, technology='NUCLEAR', target_gw=8.0,
                                                    tolerance_gw=0.001, ampl_uq_collector=None,
                                                      commissioning_cols=None, commissioning_prefix='cp_',
                                                      use_exp=True, plot=True, color=None,
                                                      forced_sample_ids=None,
                                                      forced_green_sample_ids=None,
                                                      show_legend_export=False):
        """
        Plot commissioning times for scenarios whose decided capacity matches a target.

        Steps
        -----
          1) Identify scenarios where sum(F_decided_realized_up_to) matches a target
              (either a point +/- tolerance, or an explicit interval [min, max]).
        2) Print matching scenario IDs.
        3) Retrieve commissioning columns from Samples (same source as samples.csv)
           and plot them for the selected scenarios.

        Parameters
        ----------
        technology : str
            Technology to filter in Decision_tracking (e.g., 'NUCLEAR').
        target_gw : float | tuple[float, float] | list[float]
            - If float: point target in GW (legacy behavior).
            - If tuple/list of length 2: explicit interval [min_gw, max_gw].
              Example: target_gw=(6, 8).
        tolerance_gw : float
            Used only when target_gw is a scalar:
            A scenario is kept if abs(capacity - target_gw) <= tolerance_gw.
        ampl_uq_collector : dict | None
            UQ collector dictionary. If None, use self.ampl_uq_collector.
        commissioning_cols : list[str] | None
            Optional explicit commissioning time columns to read from Samples.
            If None, columns are auto-detected with commissioning_prefix,
            then fallback patterns containing constr/construction/t_constr/cp_.
        commissioning_prefix : str
            Preferred prefix for commissioning columns in Samples.
        use_exp : bool
            If True, apply exp() to commissioning columns (log-space inputs).
        plot : bool
            If True, display figure.
        color : str | None
            Optional single color override for all traces.
            You can pass 'noir'/'black' for black, or 'gris'/'gray'/'grey' for gray.
        forced_sample_ids : list[int|str] | None
            Additional scenario IDs to always include in the plot,
            even if they do not match decided capacity target.
        forced_green_sample_ids : list[int|str] | None
            Scenario IDs to force in green in the plot.
        show_legend_export : bool
            If True, keep legend in saved PDF/PNG. If False, hide legend.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in decision_tracking.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        sample_col = 'Sample' if 'Sample' in decision_tracking.columns else decision_tracking.columns[0]
        tech_col = 'Technologies' if 'Technologies' in decision_tracking.columns else None

        if tech_col is not None:
            decision_tracking = decision_tracking.loc[decision_tracking[tech_col] == technology].copy()

        if decision_tracking.empty:
            raise ValueError(f"Aucune donnee trouvee pour la technologie : {technology}")

        cap_by_sample = (
            decision_tracking
            .assign(F_decided_realized_up_to=pd.to_numeric(decision_tracking['F_decided_realized_up_to'], errors='coerce').fillna(0.0))
            .groupby(sample_col, as_index=False)['F_decided_realized_up_to']
            .sum()
            .rename(columns={'F_decided_realized_up_to': 'Capacity_GW'})
        )

        # Selection mode:
        # - scalar target_gw => point target +/- tolerance
        # - tuple/list target_gw=(min, max) => interval filter
        is_range_target = isinstance(target_gw, (tuple, list, np.ndarray)) and len(target_gw) == 2
        if is_range_target:
            target_min = float(min(target_gw[0], target_gw[1]))
            target_max = float(max(target_gw[0], target_gw[1]))
            cap_by_sample['InTarget'] = (
                (cap_by_sample['Capacity_GW'] > target_min)
                & (cap_by_sample['Capacity_GW'] < target_max)
            )
        else:
            target_center = float(target_gw)
            target_min = float(target_center - float(tolerance_gw))
            target_max = float(target_center + float(tolerance_gw))
            cap_by_sample['InTarget'] = (
                (cap_by_sample['Capacity_GW'] > target_min)
                & (cap_by_sample['Capacity_GW'] < target_max)
            )
        selected = cap_by_sample.loc[cap_by_sample['InTarget']].copy()

        selected_ids = selected[sample_col].tolist()

        forced_sample_ids = [] if forced_sample_ids is None else list(forced_sample_ids)
        forced_green_sample_ids = [54] if forced_green_sample_ids is None else list(forced_green_sample_ids)

        selected_ids_as_str = [str(s) for s in selected_ids]
        n_in_target = len(selected_ids_as_str)
        for sid in forced_sample_ids:
            if str(sid) not in selected_ids_as_str:
                selected_ids.append(sid)
                selected_ids_as_str.append(str(sid))
        n_used_for_plot = len(selected_ids_as_str)

        print("Scenarios matching target decided capacity range:")
        print(f"Range: [{target_min}, {target_max}] GW")
        print(f"Number of scenarios in target range (strict): {n_in_target}")
        print(f"Number of scenarios used for plot (after forced additions): {n_used_for_plot}")
        print(",".join([str(s) for s in selected_ids]))
        if len(forced_sample_ids) > 0:
            print("Scenarios forces ajoutes au plot:")
            print(",".join([str(s) for s in forced_sample_ids]))

        samples = ampl_uq_collector.get('Samples')
        if not isinstance(samples, pd.DataFrame) or samples.empty:
            if os.path.exists(self.samples_file):
                samples = pd.read_csv(self.samples_file)
            else:
                raise ValueError("'Samples' introuvable dans le collecteur UQ et samples.csv non trouve.")

        # Create a robust sample key for joins with Decision_tracking scenario IDs.
        if 'Sample' not in samples.columns:
            samples['Sample'] = samples.index + 1
        sample_key = 'Sample'

        def _select_samples_by_ids(samples_df, sample_ids):
            ids_str = set(str(s) for s in sample_ids)
            out = samples_df.loc[samples_df[sample_key].astype(str).isin(ids_str)].copy()
            if not out.empty:
                return out

            sample_num = pd.to_numeric(samples_df[sample_key], errors='coerce')
            ids_num = pd.to_numeric(pd.Series(sample_ids), errors='coerce').dropna()
            if len(ids_num) == 0:
                return out

            out = samples_df.loc[sample_num.isin(ids_num.tolist())].copy()
            if not out.empty:
                return out

            # Off-by-one fallback (samples 0-based, scenario IDs 1-based).
            out = samples_df.loc[sample_num.isin((ids_num - 1).tolist())].copy()
            if not out.empty:
                out[sample_key] = pd.to_numeric(out[sample_key], errors='coerce') + 1
            return out

        if commissioning_cols is None:
            cols_by_prefix = [
                c for c in samples.columns
                if isinstance(c, str) and c.startswith(str(commissioning_prefix))
            ]
            if len(cols_by_prefix) > 0:
                commissioning_cols = cols_by_prefix
            else:
                commissioning_cols = [
                    c for c in samples.columns
                    if isinstance(c, str)
                    and re.search(r'constr|construction|t_constr|cp_', c, flags=re.IGNORECASE)
                ]

        commissioning_cols = [c for c in commissioning_cols if c in samples.columns and c != sample_key]
        if len(commissioning_cols) == 0:
            raise ValueError(
                "Aucune colonne de commissioning time detectee dans Samples. "
                "Passez commissioning_cols explicitement si necessaire."
            )

        selected_samples = _select_samples_by_ids(samples, selected_ids)

        if selected_samples.empty:
            raise ValueError("Aucun scenario a tracer n'est introuvable dans Samples.")

        for c in commissioning_cols:
            selected_samples[c] = pd.to_numeric(selected_samples[c], errors='coerce')

        if use_exp:
            selected_samples[commissioning_cols] = np.exp(selected_samples[commissioning_cols])

        long_df = selected_samples[[sample_key] + commissioning_cols].melt(
            id_vars=[sample_key],
            value_vars=commissioning_cols,
            var_name='Commissioning_Parameter',
            value_name='Commissioning_Time'
        )
        long_df.dropna(subset=['Commissioning_Time'], inplace=True)
        long_df.rename(columns={sample_key: 'Sample'}, inplace=True)

        if long_df.empty:
            raise ValueError("Aucune valeur de commissioning time exploitable apres filtrage.")

        long_df['Label'] = long_df['Commissioning_Parameter'].map(
            lambda c: self.uncert_param_meaning.get(c, c)
        )
        param_order = [self.uncert_param_meaning.get(c, c) for c in commissioning_cols]
        long_df['Label'] = pd.Categorical(long_df['Label'], categories=param_order, ordered=True)
        long_df.sort_values(by=['Sample', 'Label'], inplace=True)

        if color is None:
            default_color = self.dict_color('Electricity').get(str(technology), 'deeppink')
        else:
            user_color = str(color).strip().lower()
            default_color = 'black' if user_color == 'noir' else ('gray' if user_color in ('gris', 'grey') else str(color))

        fig = go.Figure()
        sample_order = sorted(long_df['Sample'].astype(str).unique().tolist())
        for sid in sample_order:
            trace_df = long_df.loc[long_df['Sample'].astype(str) == sid].copy()
            trace_df.sort_values(by='Label', inplace=True)
            trace_color = default_color
            if str(sid) in [str(s) for s in forced_green_sample_ids]:
                trace_color = 'green'
            trace_name = f"Scenario {sid}"
            fig.add_trace(
                go.Scatter(
                    x=trace_df['Label'].astype(str).tolist(),
                    y=trace_df['Commissioning_Time'].to_numpy(dtype=float),
                    mode='lines+markers',
                    name=trace_name,
                    line=dict(color=trace_color, width=1.5),
                    marker=dict(color=trace_color, size=6, opacity=0.75),
                    showlegend=True
                )
            )

        fig.update_layout(showlegend=True)

        tech_label = self.dict_meaning().get(technology, technology)
        if is_range_target:
            title_main = "{} - Commissioning time when decided capacity is in ]{:.2f}, {:.2f}[ GW".format(
                tech_label,
                target_min,
                target_max
            )
        else:
            title_main = "{} - Commissioning time when decided capacity is {:.2f} GW +/- {:.3f}".format(
                tech_label,
                float(target_gw),
                float(tolerance_gw)
            )
        title_with_unit = title_main
        y_min = float(long_df['Commissioning_Time'].min())
        y_max = float(long_df['Commissioning_Time'].max())
        yvals = [round(y_min, 1), round(y_max, 1)] if y_min != y_max else [round(y_min, 1)]
        xvals = [str(v) for v in param_order]
        self.custom_fig(fig, title_with_unit, yvals, xvals=xvals, type_graph='bar', y_unit='[years]')
        fig.update_layout(
            title=dict(text=title_main, x=0.5, xanchor='center'),
            template='simple_white',
            width=1200,
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Decision_tracking/Commissioning_time/"
        out_dir_raw = out_dir + "_Raw/"
        if not os.path.exists(Path(out_dir)):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(out_dir_raw)):
            Path(out_dir_raw).mkdir(parents=True, exist_ok=True)

        safe_tech = "".join(c for c in str(technology) if c.isalnum() or c in (' ', '_', '-')).strip()
        if is_range_target:
            base_name = (
                f"Commissioning_time_target_capacity_{safe_tech}_"
                f"range_{target_min:.2f}_{target_max:.2f}GW"
            )
        else:
            base_name = (
                f"Commissioning_time_target_capacity_{safe_tech}_"
                f"{float(target_gw):.2f}GW_tol_{float(tolerance_gw):.2f}GW"
            )

        fig.write_html(out_dir_raw + f"{base_name}_raw.html")

        fig_export = go.Figure(fig)
        fig_export.update_layout(
            showlegend=bool(show_legend_export),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_export.update_xaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.update_yaxes(showline=False, linecolor='rgba(0,0,0,0)')
        fig_export.write_image(out_dir + f"{base_name}.pdf", width=1200, height=550)
        fig_export.write_image(out_dir + f"{base_name}.png", width=1200, height=550)

        selected_out = cap_by_sample.loc[cap_by_sample[sample_col].astype(str).isin([str(s) for s in selected_ids])].copy()
        selected_out['Technology'] = technology
        selected_out['TargetMode'] = 'range' if is_range_target else 'point'
        selected_out['Target_GW'] = np.nan if is_range_target else float(target_gw)
        selected_out['Tolerance_GW'] = np.nan if is_range_target else float(tolerance_gw)
        selected_out['TargetMin_GW'] = target_min
        selected_out['TargetMax_GW'] = target_max
        selected_out['ForcedForPlot'] = selected_out[sample_col].astype(str).isin([str(s) for s in forced_sample_ids])
        selected_out['ForcedGreen'] = selected_out[sample_col].astype(str).isin([str(s) for s in forced_green_sample_ids])
        selected_out.to_csv(out_dir + f"{base_name}_scenarios.csv", index=False)

        long_out = long_df.copy()
        long_out['Technology'] = technology
        long_out['TargetMode'] = 'range' if is_range_target else 'point'
        long_out['Target_GW'] = np.nan if is_range_target else float(target_gw)
        long_out['Tolerance_GW'] = np.nan if is_range_target else float(tolerance_gw)
        long_out['TargetMin_GW'] = target_min
        long_out['TargetMax_GW'] = target_max
        long_out.to_csv(out_dir + f"{base_name}.csv", index=False)

        return long_out

    def graph_gwp_op_per_resource_per_year(self, ampl_uq_collector=None, plot=True,
                                            resources=None, year_start=None, year_end=None,
                                            min_mtco2=0.1):
        """
        Boxplot de la variation inter-scénarios du GWP_op par ressource et par année.
        X = années, une trace (box) par ressource colorée par dict_color('Resources').
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Gwp_breakdown' not in ampl_uq_collector:
            raise ValueError("'Gwp_breakdown' introuvable dans le collecteur UQ.")

        gwp = ampl_uq_collector['Gwp_breakdown'].copy().reset_index()
        year_col = 'Years'    if 'Years'    in gwp.columns else gwp.columns[0]
        elem_col = 'Elements' if 'Elements' in gwp.columns else gwp.columns[1]

        gwp[year_col] = gwp[year_col].astype(str).str.replace('YEAR_', '', regex=False)
        gwp['_yr']    = gwp[year_col].str.extract(r'(\d{4})')[0].astype(float)
        gwp['GWP_op'] = pd.to_numeric(gwp['GWP_op'], errors='coerce').fillna(0)
        gwp = gwp[gwp['GWP_op'] > 0]

        if year_start is not None:
            gwp = gwp[gwp['_yr'] >= float(year_start)]
        if year_end is not None:
            gwp = gwp[gwp['_yr'] <= float(year_end)]

        if resources is not None:
            res_list = [resources] if isinstance(resources, str) else list(resources)
            gwp = gwp[gwp[elem_col].isin(res_list)]

        sig = gwp.groupby(elem_col)['GWP_op'].median()
        gwp = gwp[gwp[elem_col].isin(sig[sig >= min_mtco2].index)]

        if gwp.empty:
            print("graph_gwp_op_per_resource_per_year: aucune donnée significative.")
            return gwp

        ordered_years = sorted(gwp['_yr'].dropna().unique().tolist())
        ordered_years_str = [str(int(y)) for y in ordered_years]
        res_order = (gwp.groupby(elem_col)['GWP_op'].median()
                     .sort_values(ascending=False).index.tolist())

        if not plot:
            return gwp

        color_map = self.dict_color('Resources')
        fig = go.Figure()
        for res in res_order:
            df_r = gwp[gwp[elem_col] == res]
            for yr_str in ordered_years_str:
                vals = df_r[df_r[year_col] == yr_str]['GWP_op']
                if vals.empty:
                    continue
                fig.add_trace(go.Box(
                    x=[yr_str] * len(vals),
                    y=vals,
                    name=res,
                    marker_color=color_map.get(res, '#888'),
                    showlegend=(yr_str == ordered_years_str[0]),
                    legendgroup=res,
                    boxpoints=False, notched=False,
                ))

        # ── Croix : scénario avec la plus basse consommation totale de COAL ─────
        if 'Resources' in ampl_uq_collector:
            res_df = ampl_uq_collector['Resources'].copy().reset_index()
            res_col2 = next((c for c in res_df.columns if c not in ('Sample', 'Years', 'Year', 'Res')), None)
            coal_df = res_df[res_df[res_col2] == 'COAL'].copy()
            coal_df['Res'] = pd.to_numeric(coal_df['Res'], errors='coerce').fillna(0)
            coal_total = coal_df.groupby('Sample')['Res'].sum()
            target_sample = coal_total.idxmin()
            coal_min = coal_total.min() / 1000.0
            print(f"  Scénario min COAL: sample={target_sample}, total={coal_min:.1f} TWh")
            gwp_target = gwp[gwp['Sample'] == target_sample]
            for res in res_order:
                df_r = gwp_target[gwp_target[elem_col] == res]
                for yr_str in ordered_years_str:
                    row = df_r[df_r[year_col] == yr_str]['GWP_op']
                    if row.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=[yr_str], y=[row.values[0]],
                        mode='markers',
                        marker=dict(symbol='x', size=10, color=color_map.get(res, '#888'),
                                    line=dict(width=2, color=color_map.get(res, '#888'))),
                        name=f'{res} (min COAL)',
                        legendgroup=f'cross_{res}',
                        showlegend=False,
                    ))

        first_year = ordered_years_str[0] if ordered_years_str else None
        fig.update_layout(
            title='GWP_op by resource per year — variation across scenarios',
            boxmode='group',
            xaxis=dict(categoryorder='array', categoryarray=ordered_years_str),
        )
        pio.show(fig)

        outdir = self.outdir + "TotalGWP/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/gwp_op_per_resource_per_year_raw.html")

        y_max = float(gwp['GWP_op'].max())
        yvals = [0, round(y_max, 1)]
        title_str = "<b>GWP_op by resource per year — variation across scenarios</b><br>[MtCO₂-eq]"
        self.custom_fig(fig, title_str, yvals, xvals=ordered_years_str, type_graph='bar')
        fig.update_xaxes(
            tickvals=ordered_years_str,
            ticktext=[y if (y == first_year or int(y) % 5 == 0) else '' for y in ordered_years_str],
        )
        fig.write_image(outdir + "gwp_op_per_resource_per_year.pdf", width=1400, height=550)
        plt.close()

        return gwp

    def graph_resource_usage_per_year(self, ampl_uq_collector=None, plot=True,
                                       resources=None, year_start=None, year_end=None):
        """
        Box plots de l'utilisation annuelle de chaque ressource (TWh) par année,
        sur tous les samples UQ. Similaire à graph_tech_cap mais pour les ressources.

        Parameters
        ----------
        resources : list[str] | None
            Liste de ressources à afficher. Si None, utilise dict_color('Resources').
        year_start / year_end : int | None
            Filtre sur les années.
        """
        pio.renderers.default = 'browser'

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        results = ampl_uq_collector['Resources'].copy()
        results.reset_index(inplace=True)

        if resources is not None:
            res_list = [resources] if isinstance(resources, str) else list(resources)
        else:
            res_list = list(self.dict_color('Resources').keys())

        results = results.loc[results['Resources'].astype(str).isin(res_list)].copy()
        results['Resources'] = results['Resources'].astype(str)
        results['Years'] = results['Years'].astype(str)
        results.dropna(subset=['Res'], how='all', inplace=True)
        results['Res'] = pd.to_numeric(results['Res'], errors='coerce').fillna(0) / 1000.0

        results = results.set_index(['Years', 'Resources', 'Sample'])

        df_to_plot = pd.DataFrame(index=results.index, columns=['Res'])
        for y in results.index.get_level_values('Years').unique():
            temp_y = results.loc[results.index.get_level_values('Years') == y, 'Res'].dropna(how='all')
            if not temp_y.empty:
                temp_y = self._remove_low_values(temp_y, threshold=0.01)
                df_to_plot.update(temp_y)
        df_to_plot.dropna(how='all', inplace=True)
        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '', regex=False)
        df_to_plot['Res'] = pd.to_numeric(df_to_plot['Res'], errors='coerce').fillna(0)

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        ordered_years = sorted(
            [y for y in df_to_plot['Years'].unique()
             if (year_start is None or _year_key(y) >= int(year_start))
             and (year_end   is None or _year_key(y) <= int(year_end))],
            key=_year_key
        )
        df_to_plot = df_to_plot[df_to_plot['Years'].isin(ordered_years)]

        if not plot:
            return df_to_plot

        n_years = len(ordered_years)
        tick_labels = ordered_years if n_years <= 15 else [
            y if _year_key(y) % 5 == 0 else '' for y in ordered_years
        ]

        fig = px.box(
            df_to_plot, x='Years', y='Res', color='Resources',
            title='Resource usage by year [TWh]',
            color_discrete_map=self.dict_color('Resources'),
            points='outliers', notched=False,
            category_orders={'Years': ordered_years},
        )
        fig.update_xaxes(
            categoryorder='array', categoryarray=ordered_years,
            tickmode='array', tickvals=ordered_years, ticktext=tick_labels,
        )
        pio.show(fig)

        outdir = self.outdir + "Resources/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/resource_usage_per_year_raw.html")

        y_max = float(df_to_plot['Res'].max())
        yvals = [0, round(y_max, 1)]
        title = "<b>Resource usage across scenarios</b><br>[TWh]"
        self.custom_fig(fig, title, yvals, xvals=ordered_years, type_graph='bar')
        fig.update_xaxes(
            tickmode='array', tickvals=ordered_years, ticktext=tick_labels,
        )
        fig.write_image(outdir + "resource_usage_per_year.pdf", width=1200, height=550)
        plt.close()

        return df_to_plot

    def graph_resource_total_variation(self, ampl_uq_collector=None, plot=True,
                                        resources=None, year_start=None, year_end=None):
        """
        Box plot avec les ressources en x et la somme totale sur toutes les années
        (ou la plage year_start–year_end) par sample en y.
        Montre la variation inter-scénarios de la consommation totale de chaque ressource.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        res = ampl_uq_collector['Resources'].copy().reset_index()
        res['Years'] = res['Years'].astype(str).str.replace('YEAR_', '', regex=False)
        res['Res'] = pd.to_numeric(res['Res'], errors='coerce').fillna(0) / 1000.0

        if year_start is not None:
            res = res[res['Years'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            res = res[res['Years'].apply(_year_key) <= int(year_end)]

        if resources is not None:
            res_list = [resources] if isinstance(resources, str) else list(resources)
            res = res[res['Resources'].astype(str).isin(res_list)]

        # Somme sur toutes les années par (Sample, Resource)
        df_agg = res.groupby(['Sample', 'Resources'], as_index=False)['Res'].sum()

        # Retire les ressources avec médiane nulle (insignifiantes)
        sig = df_agg.groupby('Resources')['Res'].median()
        sig_res = sig[sig > 0.1].index
        df_agg = df_agg[df_agg['Resources'].isin(sig_res)]

        # Garde uniquement les ressources avec > 10 TWh d'écart entre min et max
        spread = df_agg.groupby('Resources')['Res'].apply(lambda x: x.max() - x.min())
        keep = spread[spread > 10].index
        df_agg = df_agg[df_agg['Resources'].isin(keep)]

        # Filtre exceptionnel : garde uniquement ces ressources spécifiques
        _keep_explicit = ['GAS', 'WOOD', 'COAL', 'METHANOL_RE', 'CO2_EMISSIONS', 'CO2_CAPTURED',
                          'RES_WIND', 'RES_SOLAR', 'URANIUM']
        df_agg = df_agg[df_agg['Resources'].isin(_keep_explicit)]

        if df_agg.empty:
            print("graph_resource_total_variation: aucune donnée significative.")
            return df_agg

        # Trie les ressources par médiane décroissante
        order = df_agg.groupby('Resources')['Res'].median().sort_values(ascending=False).index.tolist()

        if not plot:
            return df_agg

        fig = px.box(
            df_agg, x='Resources', y='Res',
            color='Resources',
            title='Total resource usage across all years — variation across scenarios [TWh]',
            color_discrete_map=self.dict_color('Resources'),
            points='outliers', notched=False,
            category_orders={'Resources': order},
        )
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        pio.show(fig)

        outdir = self.outdir + "Resources/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/resource_total_variation_raw.html")

        yvals = [0, round(float(df_agg['Res'].max()), 0)]
        title_str = "<b>Total resource usage — variation across scenarios</b><br>[TWh]"
        if year_start or year_end:
            title_str = f"<b>Total resource usage ({year_start or ''}–{year_end or ''}) — variation across scenarios</b><br>[TWh]"
        self.custom_fig(fig, title_str, yvals, type_graph='bar')
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        fig.write_image(outdir + "resource_total_variation.pdf", width=1400, height=550)
        plt.close()

        return df_agg

    def graph_cop_total_variation(self, ampl_uq_collector=None, plot=True,
                                   resources=None, year_start=None, year_end=None):
        """
        Box plot avec les ressources en x et le C_op total (somme sur toutes les phases)
        par sample en y [b€]. Même principe que graph_resource_total_variation.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_op_phase_res' not in ampl_uq_collector:
            raise ValueError("'C_op_phase_res' introuvable dans le collecteur UQ.")

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        opex = ampl_uq_collector['C_op_phase_res'].copy().reset_index()
        phase_col = 'Phases' if 'Phases' in opex.columns else opex.columns[0]
        res_col   = 'Resources' if 'Resources' in opex.columns else opex.columns[1]

        opex['_year'] = opex[phase_col].astype(str).str.split('_').str[0]
        opex['C_op_phase_res'] = pd.to_numeric(opex['C_op_phase_res'], errors='coerce').fillna(0)
        opex['C_op_bEUR'] = opex['C_op_phase_res'] / 1000.0

        if year_start is not None:
            opex = opex[opex['_year'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            opex = opex[opex['_year'].apply(_year_key) <= int(year_end)]

        if resources is not None:
            res_list = [resources] if isinstance(resources, str) else list(resources)
            opex = opex[opex[res_col].astype(str).isin(res_list)]

        # Somme sur toutes les phases par (Sample, Resource)
        df_agg = opex.groupby(['Sample', res_col], as_index=False)['C_op_bEUR'].sum()
        df_agg.rename(columns={res_col: 'Resources'}, inplace=True)

        # Retire les ressources avec médiane nulle
        sig = df_agg.groupby('Resources')['C_op_bEUR'].median()
        sig_res = sig[sig.abs() > 0.001].index
        df_agg = df_agg[df_agg['Resources'].isin(sig_res)]

        if df_agg.empty:
            print("graph_cop_total_variation: aucune donnée significative.")
            return df_agg

        order = df_agg.groupby('Resources')['C_op_bEUR'].median().sort_values(ascending=False).index.tolist()

        if not plot:
            return df_agg

        fig = px.box(
            df_agg, x='Resources', y='C_op_bEUR',
            color='Resources',
            title='Total C_op by resource — variation across scenarios [b€]',
            color_discrete_map=self.dict_color('Resources'),
            points='outliers', notched=False,
            category_orders={'Resources': order},
        )
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        pio.show(fig)

        outdir = self.outdir + "Resources/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/cop_total_variation_raw.html")

        yvals = [0, round(float(df_agg['C_op_bEUR'].max()), 2)]
        title_str = "<b>Total C_op by resource — variation across scenarios</b><br>[b€]"
        self.custom_fig(fig, title_str, yvals, type_graph='bar')
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        fig.write_image(outdir + "cop_total_variation.pdf", width=1400, height=550)
        plt.close()

        return df_agg

    def graph_cop_total_variation_highlight(self, ampl_uq_collector=None, plot=True,
                                            resources=None, year_start=None, year_end=None):
        """
        Même chose que graph_cop_total_variation mais avec GAS, LFO, METHANOL_RE,
        URANIUM, COAL, H2_RE mis en évidence (couleur vive, autres en gris).
        Ticks sur l'axe y tous les 10 b€.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_op_phase_res' not in ampl_uq_collector:
            raise ValueError("'C_op_phase_res' introuvable dans le collecteur UQ.")

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        opex = ampl_uq_collector['C_op_phase_res'].copy().reset_index()
        phase_col = 'Phases' if 'Phases' in opex.columns else opex.columns[0]
        res_col   = 'Resources' if 'Resources' in opex.columns else opex.columns[1]

        opex['_year'] = opex[phase_col].astype(str).str.split('_').str[0]
        opex['C_op_phase_res'] = pd.to_numeric(opex['C_op_phase_res'], errors='coerce').fillna(0)
        opex['C_op_bEUR'] = opex['C_op_phase_res'] / 1000.0

        if year_start is not None:
            opex = opex[opex['_year'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            opex = opex[opex['_year'].apply(_year_key) <= int(year_end)]

        if resources is not None:
            res_list = [resources] if isinstance(resources, str) else list(resources)
            opex = opex[opex[res_col].astype(str).isin(res_list)]

        df_agg = opex.groupby(['Sample', res_col], as_index=False)['C_op_bEUR'].sum()
        df_agg.rename(columns={res_col: 'Resources'}, inplace=True)

        _keep = {'GAS', 'COAL', 'LFO', 'METHANOL_RE', 'AMMONIA', 'H2_RE'}
        df_agg = df_agg[df_agg['Resources'].isin(_keep)]

        sig = df_agg.groupby('Resources')['C_op_bEUR'].median()
        sig_res = sig[sig.abs() > 0.001].index
        df_agg = df_agg[df_agg['Resources'].isin(sig_res)]

        if df_agg.empty:
            print("graph_cop_total_variation_highlight: aucune donnée significative.")
            return df_agg

        order = df_agg.groupby('Resources')['C_op_bEUR'].median().sort_values(ascending=False).index.tolist()

        if not plot:
            return df_agg

        color_map = self.dict_color('Resources')

        traces = []
        for res in order:
            vals = df_agg.loc[df_agg['Resources'] == res, 'C_op_bEUR']
            traces.append(go.Box(
                y=vals, name=res,
                marker_color=color_map.get(res, '#888888'),
                boxpoints='outliers', notched=False,
            ))

        fig = go.Figure(data=traces)

        y_min = float(df_agg['C_op_bEUR'].min())
        y_max = float(df_agg['C_op_bEUR'].max())

        fig.update_layout(
            title=dict(
                text="<b>Total C_op by resource — variation across scenarios</b><br>[b€]",
                x=0.5, xanchor='center', font=dict(family='Raleway', size=22),
            ),
            yaxis=dict(
                range=[-y_max * 0.02, y_max * 1.02],
                tickvals=list(range(0, 81, 10)) + [round(y_min, 2), round(y_max, 2)],
                ticktext=[str(v) for v in range(0, 81, 10)] + [str(round(y_min, 2)), str(round(y_max, 2))],
                side='left',
                ticks='inside',
                ticklen=6,
            ),
            xaxis=dict(visible=False),
            showlegend=False,
            template='simple_white',
        )

        title_str = "<b>Total C_op by resource — variation across scenarios</b><br>[b€]"
        yvals = [0, round(float(df_agg['C_op_bEUR'].max()), 2)]
        self.custom_fig(fig, title_str, yvals, type_graph='bar')
        pio.show(fig)

        outdir = self.outdir + "Resources/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/cop_total_variation_highlight_raw.html")
        fig.write_image(outdir + "cop_total_variation_highlight.pdf", width=1400, height=550)
        plt.close()

        return df_agg

    def graph_cinv_total_by_sector(self, ampl_uq_collector=None, plot=True,
                                    year_start=None, year_end=None):
        """
        Box plot du C_inv total (somme sur toutes les phases) par secteur et par sample [b€].
        Les secteurs correspondent aux catégories de dict_color.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'C_inv_phase_tech' not in ampl_uq_collector:
            raise ValueError("'C_inv_phase_tech' introuvable dans le collecteur UQ.")

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        # ── Mapping technologie → secteur ──────────────────────────────────────────
        sectors = ['Electricity', 'Heat_low_T', 'Heat_high_T', 'Mobility', 'Freight',
                   'Ammonia', 'Methanol', 'HVC', 'Conversion', 'Storage']
        sector_colors = {
            'Electricity': 'dodgerblue', 'Heat_low_T': 'indianred',
            'Heat_high_T': 'red', 'Mobility': 'goldenrod', 'Freight': 'darkgoldenrod',
            'Ammonia': 'slateblue', 'Methanol': 'orchid', 'HVC': 'cyan',
            'Conversion': 'mediumpurple', 'Storage': 'chartreuse',
        }
        tech_to_sector = {}
        for s in sectors:
            for tech in self.dict_color(s).keys():
                tech_to_sector[tech] = s

        # ── C_inv_phase_tech ───────────────────────────────────────────────────────
        inv = ampl_uq_collector['C_inv_phase_tech'].copy().reset_index()
        phase_col = 'Phases'      if 'Phases'      in inv.columns else inv.columns[0]
        tech_col  = 'Technologies' if 'Technologies' in inv.columns else inv.columns[1]

        inv['_year'] = inv[phase_col].astype(str).str.split('_').str[0]
        inv['C_inv_phase_tech'] = pd.to_numeric(inv['C_inv_phase_tech'], errors='coerce').fillna(0)
        inv['C_inv_bEUR'] = inv['C_inv_phase_tech'] / 1000.0

        if year_start is not None:
            inv = inv[inv['_year'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            inv = inv[inv['_year'].apply(_year_key) <= int(year_end)]

        inv['Sector'] = inv[tech_col].astype(str).map(tech_to_sector)
        inv = inv.dropna(subset=['Sector'])

        df_agg = inv.groupby(['Sample', 'Sector'], as_index=False)['C_inv_bEUR'].sum()

        # ── Soustraire le coût de retour (valeur résiduelle après 2050) ───────────
        if 'Cost_return' in ampl_uq_collector:
            ret = ampl_uq_collector['Cost_return'].copy().reset_index()
            ret_tech_col = 'Technologies' if 'Technologies' in ret.columns else ret.columns[1]
            ret['C_inv_return'] = pd.to_numeric(ret['C_inv_return'], errors='coerce').fillna(0)
            ret['C_ret_bEUR'] = ret['C_inv_return'] / 1000.0
            ret['Sector'] = ret[ret_tech_col].astype(str).map(tech_to_sector)
            ret = ret.dropna(subset=['Sector'])
            ret_agg = ret.groupby(['Sample', 'Sector'], as_index=False)['C_ret_bEUR'].sum()
            df_agg = df_agg.merge(ret_agg, on=['Sample', 'Sector'], how='left')
            df_agg['C_ret_bEUR'] = df_agg['C_ret_bEUR'].fillna(0)
            df_agg['C_inv_bEUR'] = df_agg['C_inv_bEUR'] - df_agg['C_ret_bEUR']
            df_agg.drop(columns=['C_ret_bEUR'], inplace=True)

        # Filtre secteurs insignifiants
        sig = df_agg.groupby('Sector')['C_inv_bEUR'].median()
        sig_sec = sig[sig.abs() > 0.001].index
        df_agg = df_agg[df_agg['Sector'].isin(sig_sec)]

        if df_agg.empty:
            print("graph_cinv_total_by_sector: aucune donnée significative.")
            return df_agg

        order = df_agg.groupby('Sector')['C_inv_bEUR'].median().sort_values(ascending=False).index.tolist()

        if not plot:
            return df_agg

        fig = px.box(
            df_agg, x='Sector', y='C_inv_bEUR',
            color='Sector',
            title='Total C_inv by sector — variation across scenarios [b€]',
            color_discrete_map=sector_colors,
            points='outliers', notched=False,
            category_orders={'Sector': order},
        )

        fig.update_traces(
            boxpoints='outliers',
            marker=dict(size=6, opacity=0.85),
            line=dict(width=1.6),
        )

        y_max = float(df_agg['C_inv_bEUR'].max())
        y_pad = max(y_max * 0.12, 0.25)
        pdf_width = max(900, len(order) * 145)

        fig.update_layout(
            template='simple_white',
            font=dict(family='Arial', size=16),
            title=dict(
                text='Total C_inv by sector — variation across scenarios',
                x=0.5,
                xanchor='center',
                font=dict(family='Arial', size=24),
            ),
            width=pdf_width,
            height=550,
            margin=dict(l=60, r=20, t=90, b=130),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title='',
                tickangle=0,
                tickfont=dict(size=16),
                categoryorder='array',
                categoryarray=order,
                showgrid=False,
                showline=True,
                linecolor='rgba(90,90,90,0.9)',
            ),
            yaxis=dict(
                title='[b€]',
                range=[0, y_max + y_pad],
                tickfont=dict(size=16),
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor='rgba(90,90,90,0.9)',
                ticks='inside',
                ticklen=8,
            ),
        )

        pio.show(fig)

        outdir = self.outdir + "CapexOpex/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/cinv_total_by_sector_raw.html")

        fig.write_image(outdir + "cinv_total_by_sector.pdf", width=pdf_width, height=550)
        plt.close()

        return df_agg

    def graph_electrofuels(self, ampl_uq_collector = None, plot = True):
        pio.renderers.default = 'browser'
        
        if ampl_uq_collector == None:
            ampl_uq_collector = self.ampl_uq_collector
        
        col_plot = ['METHANOL_RE','AMMONIA_RE','GAS_RE','H2_RE']
        results = ampl_uq_collector['Resources'].copy()
        results.reset_index(inplace=True)
        results = results.loc[results['Resources'].isin(col_plot),:]
        results['Resources'] = results['Resources'].astype("str")
        results.dropna(how='all',inplace=True)
        results['Res']/=1000
        
        df_to_plot = results.copy()
        df_to_plot = df_to_plot.set_index(['Years','Resources','Sample'])
        df_to_plot.dropna(how='all',inplace=True)
        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '')
        
            
        if plot:
            fig = px.box(df_to_plot, x='Years', color='Resources',y='Res',
                           title='Electrofuels [TWh]',
                           color_discrete_map=self.color_dict_full,notched=False,
                           points = 'outliers')
                
            fig.update_xaxes(categoryorder='array', categoryarray= sorted(df_to_plot['Years'].unique()))
            pio.show(fig)
            
            fig.write_html(self.outdir+"/Electrofuels.html")
        
            title = "Imported renewable electrofuels"
            yvals = [0,round(max(df_to_plot['Res']))]
            
            self.custom_fig(fig, title, yvals, type_graph='bar', y_unit='[TWh]')

            fig.write_image(self.outdir+"Electrofuels_no_outlier.pdf", width=1200, height=550)
            plt.close()
    
    def graph_local_RE(self, ampl_uq_collector = None, plot = True):
        pio.renderers.default = 'browser'
        
        if ampl_uq_collector == None:
            ampl_uq_collector = self.ampl_uq_collector
        
        col_plot = ['PV_RESIDENTIAL', 'PV_FIELD','WIND_ONSHORE','WIND_OFFSHORE']
        results = ampl_uq_collector['Assets'].copy()
        results.reset_index(inplace=True)
        results = results.loc[results['Technologies'].isin(col_plot),:]
        results['Technologies'] = results['Technologies'].astype("str")
        results.dropna(how='all',inplace=True)
        results['F_year']/=1000
        
        df_to_plot = results.copy()
        df_to_plot = df_to_plot.set_index(['Years','Technologies','Sample'])
        df_to_plot.dropna(how='all',inplace=True)
        df_to_plot = self._fill_df_to_plot_w_zeros(df_to_plot)
        df_to_plot.reset_index(inplace=True)
        
        df_to_plot['Years'] = df_to_plot['Years'].str.replace('YEAR_', '')
        
            
        if plot:
            fig = px.box(df_to_plot, x='Years', color='Technologies',y='F',
                           title='Local renewable capacities [GW]',
                           color_discrete_map=self.color_dict_full,notched=False,
                           points='outliers')
                
            fig.update_xaxes(categoryorder='array', categoryarray= sorted(df_to_plot['Years'].unique()))
            pio.show(fig)
            
            fig.write_html(self.outdir+"/Local_Ren_Cap.html")
        
            title = "Local renewables - Capacities"
            yvals = [0,round(max(df_to_plot['F']))]
            
            self.custom_fig(fig, title, yvals, type_graph='bar', y_unit='[GW]')

            fig.write_image(self.outdir+"Local_Ren_Cap.pdf", width=1200, height=550)
            plt.close()
            
            fig = px.box(df_to_plot, x='Years', color='Technologies',y='F_year',
                           title='Local renewable production [TWh]',
                           color_discrete_map=self.color_dict_full,notched=False,
                           points='outliers')
                
            fig.update_xaxes(categoryorder='array', categoryarray= sorted(df_to_plot['Years'].unique()))
            pio.show(fig)
            
            fig.write_html(self.outdir+"/Local_Ren_Prod.html")
        
            title = "Local renewables - Production"
            yvals = [0,round(max(df_to_plot['F_year']))]
            
            self.custom_fig(fig, title, yvals, type_graph='bar', y_unit='[TWh]')

            fig.write_image(self.outdir+"Local_Ren_Prod.pdf", width=1200, height=550)
            plt.close()
    
    
    
    
    def graph_layer(self, ampl_uq_collector = None, plot = True):
        pio.renderers.default = 'browser'
        
        if ampl_uq_collector == None:
            ampl_uq_collector = self.ampl_uq_collector
        
        col_plot = ['METHANOL','AMMONIA','ELECTRICITY','GAS','H2','WOOD','WET_BIOMASS','HEAT_HIGH_T',
                'HEAT_LOW_T_DECEN','HEAT_LOW_T_DHN','HVC',
                'MOB_FREIGHT_BOAT','MOB_FREIGHT_RAIL','MOB_FREIGHT_ROAD','MOB_PRIVATE',
                'MOB_PUBLIC','Sample']
        results = ampl_uq_collector['Year_balance'].copy()
        results = results[col_plot]
        results.reset_index(inplace=True)
        results = results.set_index(['Years','Elements','Sample'])
        df_to_plot_full = dict.fromkeys(col_plot)
        for k in results.columns:
            df_to_plot = pd.DataFrame(index=results.index,columns=[k])
            temp = results.loc[:,k].dropna(how='all')
            for y in results.index.get_level_values(0).unique():
                temp_y = temp.loc[temp.index.get_level_values('Years') == y,:] 
                if not temp_y.empty:
                    temp_y = self._remove_low_values(temp_y, threshold=0.01)
                    df_to_plot.update(temp_y)
            df_to_plot.dropna(how='all',inplace=True)
            
            
            df_to_plot_prod = df_to_plot.loc[df_to_plot[k]>0]
            df_to_plot_cons = df_to_plot.loc[df_to_plot[k]<0]
            
            df_to_plot_prod = self._fill_df_to_plot_w_zeros(df_to_plot_prod)
            df_to_plot_cons = self._fill_df_to_plot_w_zeros(df_to_plot_cons)
            
            df_to_plot_prod.reset_index(inplace=True)
            df_to_plot_cons.reset_index(inplace=True)
            
            df_to_plot_cons.loc[df_to_plot_cons[k]<0,k] = - df_to_plot_cons.loc[df_to_plot_cons[k]<0,k]
            
            
            df_to_plot_prod['Elements'] = df_to_plot_prod['Elements'].astype("str")
            df_to_plot_prod['Years'] = df_to_plot_prod['Years'].str.replace('YEAR_', '')
            df_to_plot_prod[k] /= 1000
            
            df_to_plot_cons['Elements'] = df_to_plot_cons['Elements'].astype("str")
            df_to_plot_cons['Years'] = df_to_plot_cons['Years'].str.replace('YEAR_', '')
            df_to_plot_cons[k] /= 1000

            
            
            if plot:
                fig_prod = px.box(df_to_plot_prod, x='Years', color='Elements',y=k,
                               title='{} - Production'.format(k),
                               color_discrete_map=self.color_dict_full,notched=False,
                               points='outliers')
                    
                fig_prod.update_xaxes(categoryorder='array', categoryarray= sorted(df_to_plot_prod['Years'].unique()))
                pio.show(fig_prod)
                
                if not os.path.exists(Path(self.outdir+"Layers")):
                    Path(self.outdir+"Layers").mkdir(parents=True,exist_ok=True)
                
                if not os.path.exists(Path(self.outdir+"Layers/_Raw/")):
                    Path(self.outdir+"Layers/_Raw").mkdir(parents=True,exist_ok=True)
                    
                fig_prod.write_html(self.outdir+"Layers/_Raw/"+k+"_Prod.html")
                
                fig_cons = px.box(df_to_plot_cons, x='Years', color='Elements',y=k,
                               title='{} - Consumption'.format(k),
                               color_discrete_map=self.color_dict_full,notched=False,
                               points='outliers')
                    
                fig_cons.update_xaxes(categoryorder='array', categoryarray= sorted(df_to_plot_cons['Years'].unique()))
                pio.show(fig_cons)
                fig_cons.write_html(self.outdir+"Layers/_Raw/"+k+"_Cons.html")
            
                title = "{} - Supply".format(k)
                yvals = [0,round(max(df_to_plot_prod[k]))]
                
                self.custom_fig(fig_prod, title, yvals, type_graph='bar', y_unit='[TWh]')

                fig_prod.write_image(self.outdir+"Layers/"+k+" - Production.pdf", width=1200, height=550)
                plt.close()
                
                title = "{} - Consumption".format(k)
                yvals = [0,round(max(df_to_plot_cons[k]))]
                
                self.custom_fig(fig_cons, title, yvals, type_graph='bar', y_unit='[TWh]')

                fig_cons.write_image(self.outdir+"Layers/"+k+" - Consumption.pdf", width=1200, height=550)
                plt.close()
                

            df_to_plot_full[k] = df_to_plot
            
        return df_to_plot_full


    def graph_decision_timing_ridge(self, ampl_uq_collector=None, plot=True, use_phase_increment=True):
        """
        Ridge plot of decision timing based on F_decided_realized_up_to.

        The plot shows, for each electricity technology, how the share (%) of
        decided capacity is distributed across phases after aggregating all UQ scenarios.
        """

        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy()
        decision_tracking = decision_tracking.reset_index()

        if 'F_decided_realized_up_to' not in decision_tracking.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col = 'Phases' if 'Phases' in decision_tracking.columns else decision_tracking.columns[0]
        tech_col = 'Technologies' if 'Technologies' in decision_tracking.columns else decision_tracking.columns[1]
        sample_col = 'Sample'
        if sample_col not in decision_tracking.columns:
            decision_tracking[sample_col] = 0
        n_scenarios = int(decision_tracking[sample_col].nunique())

        decision_tracking = decision_tracking[[sample_col, phase_col, tech_col, 'F_decided_realized_up_to']].copy()
        decision_tracking.rename(columns={'F_decided_realized_up_to': 'Decision_GW'}, inplace=True)

        elec_techs = self._get_electricity_technologies()
        decision_tracking = decision_tracking.loc[decision_tracking[tech_col].isin(elec_techs)].copy()
        if decision_tracking.empty:
            raise ValueError("Aucune donnée de décision trouvée pour les technologies électriques.")

        phase_order = sorted(decision_tracking[phase_col].dropna().unique().tolist(), key=self._phase_sort_key)
        phase_order_map = {phase: i for i, phase in enumerate(phase_order)}

        grouped = (
            decision_tracking
            .groupby([sample_col, tech_col, phase_col], as_index=False)['Decision_GW']
            .sum()
        )

        grouped['__phase_order'] = grouped[phase_col].map(phase_order_map)
        grouped.sort_values(by=[sample_col, tech_col, '__phase_order'], inplace=True)

        if use_phase_increment:
            grouped['Decision_GW'] = grouped.groupby([sample_col, tech_col])['Decision_GW'].diff().fillna(grouped['Decision_GW'])
            grouped['Decision_GW'] = grouped['Decision_GW'].clip(lower=0)

        aggregated = grouped.groupby([tech_col, phase_col], as_index=False)['Decision_GW'].sum()

        tech_order = [tech for tech in elec_techs if tech in set(aggregated[tech_col])]
        full_index = pd.MultiIndex.from_product([tech_order, phase_order], names=[tech_col, phase_col])
        full_df = pd.DataFrame(index=full_index).reset_index()
        full_df = full_df.merge(aggregated, on=[tech_col, phase_col], how='left')
        full_df['Decision_GW'] = full_df['Decision_GW'].fillna(0)

        total_display_threshold_gw = 0.001
        full_df['Total_GW'] = full_df.groupby(tech_col)['Decision_GW'].transform('sum')
        full_df = full_df.loc[full_df['Total_GW'] >= total_display_threshold_gw].copy()
        full_df['Share_pct'] = 100 * full_df['Decision_GW'] / full_df['Total_GW']

        tech_order = [tech for tech in tech_order if tech in set(full_df[tech_col])]
        if not tech_order:
            raise ValueError("Aucune technologie électrique avec un total décidé >= 0.001 GW à tracer.")

        x_centers = np.array([self._phase_center_numeric(phase) for phase in phase_order], dtype=float)
        if len(np.unique(x_centers)) == 1:
            x_centers = np.array([0.0 for _ in phase_order], dtype=float)
            bandwidth = 0.35
        else:
            spacing = np.diff(np.sort(np.unique(x_centers)))
            bandwidth = max(0.25, 0.35 * float(np.median(spacing)))

        x_min = float(x_centers.min() - 0.8)
        x_max = float(x_centers.max() + 0.8)
        x_grid = np.linspace(x_min, x_max, 600)

        electricity_colors = self.dict_color('Electricity')
        fallback_colors = px.colors.qualitative.Safe
        fig = go.Figure()

        for idx, tech in enumerate(tech_order):
            temp = full_df.loc[full_df[tech_col] == tech, [phase_col, 'Share_pct']].copy()
            temp = temp.set_index(phase_col).reindex(phase_order).fillna(0).reset_index()

            weights = temp['Share_pct'].to_numpy(dtype=float) / 100.0
            if np.isclose(weights.sum(), 0):
                continue

            density = np.zeros_like(x_grid)
            for mu, w in zip(x_centers, weights):
                density += w * np.exp(-0.5 * ((x_grid - mu) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

            ridge_height = 0.95
            if np.nanmax(density) > 0:
                density_scaled = density / np.nanmax(density) * ridge_height
            else:
                density_scaled = np.zeros_like(density)

            y0 = float(idx)
            y_curve = y0 + density_scaled

            x_fill = np.concatenate([x_grid, x_grid[::-1]])
            y_fill = np.concatenate([np.full_like(x_grid, y0), y_curve[::-1]])

            color = self.color_dict_full.get(
                tech,
                electricity_colors.get(tech, fallback_colors[idx % len(fallback_colors)])
            )

            fig.add_trace(
                go.Scatter(
                    x=x_fill,
                    y=y_fill,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.35)', width=1.0),
                    fill='toself',
                    fillcolor=color,
                    opacity=0.55,
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

            fig.add_shape(
                type='line',
                x0=x_min,
                x1=x_max,
                y0=y0,
                y1=y0,
                line=dict(color='rgba(0,0,0,0.65)', width=1)
            )

        meaning = self.dict_meaning()
        tech_labels = [meaning.get(tech, tech) for tech in tech_order]
        phase_labels = [str(phase).replace('_', '-') for phase in phase_order]

        fig.update_layout(
            title="<b>Décisions d'investissement agrégées (F_decided_realized_up_to)</b><br>Part de capacité décidée par phase [%]",
            template='simple_white',
            width=1300,
            height=max(700, 42 * len(tech_order) + 180),
            margin=dict(l=200, r=30, t=90, b=70),
            showlegend=False,
            xaxis=dict(
                title='Phases',
                tickmode='array',
                tickvals=x_centers,
                ticktext=phase_labels,
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                title='Technologies électriques',
                tickmode='array',
                tickvals=list(range(len(tech_order))),
                ticktext=tech_labels,
                showgrid=False,
                zeroline=False,
                range=[-0.6, len(tech_order) - 0.05],
                automargin=True,
            )
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Decision_tracking/")):
            Path(self.outdir + "Decision_tracking/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Decision_tracking/_Raw/")):
            Path(self.outdir + "Decision_tracking/_Raw/").mkdir(parents=True, exist_ok=True)

        method_label = 'increment' if use_phase_increment else 'cumulative'
        fig.write_html(self.outdir + f"Decision_tracking/_Raw/F_decided_realized_up_to_ridge_{method_label}.html")
        fig.write_image(self.outdir + f"Decision_tracking/F_decided_realized_up_to_ridge_{method_label}.pdf", width=1300, height=max(700, 42 * len(tech_order) + 180))

        csv_out = full_df.rename(columns={phase_col: 'Phase', tech_col: 'Technology'})
        csv_out.to_csv(self.outdir + f"Decision_tracking/F_decided_realized_up_to_share_{method_label}.csv", index=False)

        return csv_out


    def graph_decision_timing_heatmap(self, ampl_uq_collector=None, plot=True, show_legend_export=True,
                                      zero_display_tolerance=0.005, technology=None,
                                      exclude_technologies=None, include_ccgt=False,
                                      min_capacity_gw=0.1,
                                      tech_order=None):
        """
        Heatmap du timing des décisions basée sur F_decided_realized_up_to.

        Cette vue affiche les mêmes données agrégées que le ridge (Share_pct),
        mais sous forme de carte de chaleur par phase et technologie.

        Parameters
        ----------
        show_legend_export : bool
            If True, keep colorbar in saved PDF/PNG. If False, hide it in saved files.
        zero_display_tolerance : float
            Shares below this value (in [0,1]) are forced to 0 for display.
            Default 0.005 keeps consistency with hover rounding to 2 decimals.
        technology : str | None
            If provided, filter the heatmap to a single electricity technology
            (e.g., 'NUCLEAR'). If None, keep all electricity technologies.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector
    
        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        samples_df = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples_df.columns:
            samples_df['Sample'] = np.arange(1, len(samples_df) + 1)

        # Capacité décidée totale pour NUCLEAR par scénario
        nuclear = dt[dt['Technologies'] == 'NUCLEAR']
        cap = nuclear.groupby('Sample')['F_decided_realized_up_to'].sum().reset_index()
        cap.columns = ['Sample', 'Total_GW']

        # Scénarios où ce n'est pas 8 GW
        not_8 = cap[cap['Total_GW'].round(3) != 8.0]

        # Commissioning time de nuclear
        cp_col = 'cp_NUCLEAR' if 'cp_NUCLEAR' in samples_df.columns else None
        if cp_col:
            ct = samples_df[['Sample', cp_col]].copy()
            ct[cp_col] = np.ceil(np.exp(ct[cp_col]))
            not_8 = not_8.merge(ct, on='Sample', how='left')

        print(f"{len(not_8)} scénarios avec capacité NUCLEAR ≠ 8 GW :")
        print(not_8.to_string(index=False))

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy()
        decision_tracking = decision_tracking.reset_index()

        if 'F_decided_realized_up_to' not in decision_tracking.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col = 'Phases' if 'Phases' in decision_tracking.columns else decision_tracking.columns[0]
        tech_col = 'Technologies' if 'Technologies' in decision_tracking.columns else decision_tracking.columns[1]
        sample_col = 'Sample'
        if sample_col not in decision_tracking.columns:
            decision_tracking[sample_col] = 0
        n_scenarios = int(decision_tracking[sample_col].nunique())

        decision_tracking = decision_tracking[[sample_col, phase_col, tech_col, 'F_decided_realized_up_to']].copy()
        decision_tracking.rename(columns={'F_decided_realized_up_to': 'Decision_GW'}, inplace=True)

        elec_techs = self._get_electricity_technologies()
        _default_exclude = {'HYDRO_RIVER', 'COAL_IGCC', 'COAL_US', 'CCGT_AMMONIA'}
        _user_exclude = set(exclude_technologies) if exclude_technologies else set()
        if not include_ccgt:
            _user_exclude |= {t for t in elec_techs if 'CCGT' in t.upper()}
        elec_techs = [t for t in elec_techs if t not in _default_exclude | _user_exclude]
        decision_tracking = decision_tracking.loc[decision_tracking[tech_col].isin(elec_techs)].copy()
        if decision_tracking.empty:
            raise ValueError("Aucune donnée de décision trouvée pour les technologies électriques.")

        if technology is not None:
            selected_tech = str(technology)
            if selected_tech not in elec_techs:
                raise ValueError(f"Technologie électrique inconnue: {selected_tech}")
            decision_tracking = decision_tracking.loc[decision_tracking[tech_col] == selected_tech].copy()
            if decision_tracking.empty:
                raise ValueError(f"Aucune donnée de décision trouvée pour la technologie {selected_tech}.")

        phase_order = sorted(decision_tracking[phase_col].dropna().unique().tolist(), key=self._phase_sort_key)
        phase_order_map = {phase: i for i, phase in enumerate(phase_order)}

        grouped = (
            decision_tracking
            .groupby([sample_col, tech_col, phase_col], as_index=False)['Decision_GW']
            .sum()
        )

        grouped['__phase_order'] = grouped[phase_col].map(phase_order_map)
        grouped.sort_values(by=[sample_col, tech_col, '__phase_order'], inplace=True)

        # Option A : fraction of scenarios with a non-zero decision in each phase.
        # On divise par n_scenarios (pas par le nb de lignes présentes) pour inclure
        # les scénarios qui ne décident pas du tout la technologie.
        decision_threshold_gw = 0.001  # 1 MW
        grouped['has_decision'] = (grouped['Decision_GW'] > decision_threshold_gw).astype(float)
        aggregated = grouped.groupby([tech_col, phase_col], as_index=False)['has_decision'].sum()
        aggregated['Share_pct'] = aggregated['has_decision'] / n_scenarios
        aggregated.drop(columns='has_decision', inplace=True)

        _default_tech_order = [
            'NUCLEAR', 'WIND_ONSHORE', 'NUCLEAR_SMR', 'PV_RESIDENTIAL',
            'PV_FIELD', 'GEOTHERMAL', 'WIND_OFFSHORE',
        ]
        _preferred = tech_order if tech_order is not None else _default_tech_order
        available = set(aggregated[tech_col])
        tech_order = [t for t in _preferred if t in available] + \
                     [t for t in elec_techs if t in available and t not in _preferred]
        full_index = pd.MultiIndex.from_product([tech_order, phase_order], names=[tech_col, phase_col])
        full_df = pd.DataFrame(index=full_index).reset_index()
        full_df = full_df.merge(aggregated, on=[tech_col, phase_col], how='left')
        full_df['Share_pct'] = full_df['Share_pct'].fillna(0)

        """
        # Exclure les technos qui ne font jamais de décision dans aucun scénario.
        full_df['Max_share'] = full_df.groupby(tech_col)['Share_pct'].transform('max')
        full_df = full_df.loc[full_df['Max_share'] > 0].copy()
        """
        tech_order = [tech for tech in tech_order if tech in set(full_df[tech_col])]
        if not tech_order:
            raise ValueError("Aucune technologie électrique avec un total decide >= 1 MW a tracer.")

        matrix = (
            full_df
            .pivot(index=tech_col, columns=phase_col, values='Share_pct')
            .reindex(index=tech_order, columns=phase_order)
            .fillna(0)
        )
        matrix = matrix.mask(matrix < float(zero_display_tolerance), 0.0)

        meaning = self.dict_meaning()
        gray = 'rgb(90,90,90)'

        # X-axis: use end year of each phase as short label (e.g. "2025-2026" → "2026")
        def _phase_start_year(phase_str):
            parts = str(phase_str).replace('_', '-').split('-')
            nums = [p for p in parts if p.isdigit() and len(p) == 4]
            return nums[0] if nums else str(phase_str)

        phase_end_years = [_phase_start_year(p) for p in phase_order]
        _last_year = phase_end_years[-1] if phase_end_years else None
        x_tickvals_shown = [y for y in phase_end_years if (y.isdigit() and int(y) % 5 == 0) or y == _last_year]
        x_ticktext = phase_end_years

        # Min / median / max decided capacity per technology across scenarios (sum over phases per scenario).
        tech_stats = (
            grouped
            .groupby([sample_col, tech_col], as_index=False)['Decision_GW']
            .sum()
            .groupby(tech_col)['Decision_GW']
            .agg(['min', 'median', 'max'])
            .reindex(matrix.index)
            .fillna(0.0)
        )
        def _fmt_gw(v):
            s = f"{v:.1f}"
            return s[:-2] if s.endswith('.0') else s

        tech_labels = [
            meaning.get(tech, tech)
            for tech in matrix.index
        ]

        # Médiane par (tech, phase) — uniquement sur les scénarios qui décident
        median_phase = (
            grouped[grouped['Decision_GW'] > 0.001]
            .groupby([tech_col, phase_col])['Decision_GW']
            .median()
            .reset_index()
            .pivot(index=tech_col, columns=phase_col, values='Decision_GW')
            .reindex(index=matrix.index, columns=phase_order)
            .fillna(0.0)
        )
        small_cells = (median_phase < min_capacity_gw) | (matrix == 0)
        matrix = matrix.where(~small_cells, 0.0)
        text_matrix = median_phase.where(~small_cells).applymap(
            lambda v: _fmt_gw(v) if pd.notna(v) else ""
        )

        # Normalisation par ligne : chaque ligne est mise à l'échelle sur son propre max.
        # Les couleurs reflètent l'intensité relative au sein de chaque technologie,
        # indépendamment des autres lignes. Les valeurs réelles restent dans customdata/texte.
        norm_matrix = matrix.copy().astype(float)
        for _i in range(len(norm_matrix)):
            _row_max = norm_matrix.iloc[_i].max()
            if _row_max > 0:
                norm_matrix.iloc[_i] /= _row_max

        fig = go.Figure(
            data=go.Heatmap(
                z=norm_matrix.values,
                x=phase_end_years,
                y=tech_labels,
                text=text_matrix.values,
                texttemplate='%{text}',
                textfont=dict(size=9, color='black'),
                customdata=matrix.values,
                zmin=0,
                zmax=1,
                colorscale=[
                    [0.00, 'rgb(247,251,255)'],
                    [0.15, 'rgb(198,219,239)'],
                    [0.35, 'rgb(107,174,214)'],
                    [0.60, 'rgb(49,130,189)'],
                    [0.80, 'rgb(22,96,162)'],
                    [1.00, 'rgb(8,48,107)'],
                ],
                colorbar=dict(
                    title=dict(text='% of scenarios', font=dict(size=13, color=gray)),
                    tickmode='array',
                    tickvals=[0, 0.5, 1],
                    ticktext=['0%', '50%', '100%'],
                    tickfont=dict(size=12, color=gray),
                    thickness=14,
                    len=0.8,
                ),
                hovertemplate='%{y}<br>Year: %{x}<br>Scenarios: %{customdata:.0%}<extra></extra>'
            )
        )

        cell_px = 46
        n_cols = max(1, len(phase_end_years))
        n_rows = max(1, len(tech_order))
        left_margin = 260
        right_margin = 110
        top_margin = 80
        bottom_margin = 55
        plot_width = max(900, left_margin + right_margin + cell_px * n_cols)
        plot_height = max(300, top_margin + bottom_margin + cell_px * n_rows)

        title_text = (
            f"Timing of capacity decisions — {meaning.get(str(technology), technology)} ({n_scenarios} scenarios)"
            if technology is not None
            else f"Timing of capacity decisions ({n_scenarios} scenarios)"
        )

        fig.update_layout(
            title=dict(text=''),
            template='simple_white',
            width=plot_width,
            height=plot_height,
            margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
            font=dict(family='Rawline', color=gray, size=12),
            xaxis=dict(
                automargin=False,
                tickmode='array',
                tickvals=x_tickvals_shown,
                ticktext=x_tickvals_shown,
                tickangle=0,
                tickfont=dict(size=20, color=gray),
                tickcolor=gray,
                ticks='outside',
                ticklen=4,
            ),
            yaxis=dict(
                automargin=False,
                categoryorder='array',
                categoryarray=tech_labels,
                autorange='reversed',
                tickfont=dict(size=20, color='black'),
                ticks='',
            ),
        )

        if plot:
            pio.show(fig)

        if not os.path.exists(Path(self.outdir + "Decision_tracking/")):
            Path(self.outdir + "Decision_tracking/").mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Path(self.outdir + "Decision_tracking/_Raw/")):
            Path(self.outdir + "Decision_tracking/_Raw/").mkdir(parents=True, exist_ok=True)

        method_label = 'increment'
        tech_suffix = ""
        if technology is not None:
            safe_tech = "".join(c for c in str(technology) if c.isalnum() or c in (' ', '_', '-')).strip()
            tech_suffix = f"_{safe_tech}"

        fig.write_html(self.outdir + f"Decision_tracking/_Raw/F_decided_realized_up_to_heatmap_{method_label}{tech_suffix}.html")

        # Export PDF/SVG via matplotlib (fiable, sans Kaleido).
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list('decision', [
            '#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'
        ])

        mat = matrix.values
        n_r, n_c = mat.shape
        cell_w = 0.42
        cell_h = 0.95          # plus haut → remplit mieux une page paysage
        fig_w = max(10, n_c * cell_w + 4.5)
        fig_h = max(4,  n_r * cell_h + 1.5)

        # Normalisation par ligne pour matplotlib (même logique que plotly).
        norm_mat = mat.astype(float).copy()
        for _i in range(n_r):
            _row_max = norm_mat[_i].max()
            if _row_max > 0:
                norm_mat[_i] /= _row_max

        fig_mpl, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(norm_mat, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest')

        # Axes.
        _shown_set = set(x_tickvals_shown)
        _x_tick_indices = [i for i, y in enumerate(phase_end_years) if y in _shown_set]
        _x_tick_labels  = [phase_end_years[i] for i in _x_tick_indices]
        ax.set_xticks(_x_tick_indices)
        ax.set_xticklabels(_x_tick_labels, fontsize=20, color='#5a5a5a', rotation=0)
        ax.set_yticks(range(n_r))
        ax.set_yticklabels(tech_labels, fontsize=20, color='black')
        ax.tick_params(length=0)

        text_vals = text_matrix.values
        for row in range(n_r):
            for col in range(n_c):
                txt = text_vals[row, col]
                if txt:
                    txt_color = 'white' if norm_mat[row, col] > 0.5 else 'black'
                    ax.text(col, row, txt, ha='center', va='center',
                            fontsize=12, color=txt_color, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colorbar.
        if show_legend_export:
            cbar = fig_mpl.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['0%', '50%', '100%'])
            cbar.ax.tick_params(labelsize=9, colors='#5a5a5a')
            cbar.outline.set_visible(False)
            cbar.set_label('% of scenarios', fontsize=10, color='#5a5a5a')

        fig_mpl.tight_layout()

        pdf_path = self.outdir + f"Decision_tracking/F_decided_realized_up_to_heatmap_{method_label}{tech_suffix}.pdf"
        svg_path = self.outdir + f"Decision_tracking/F_decided_realized_up_to_heatmap_{method_label}{tech_suffix}.svg"
        fig_mpl.savefig(pdf_path, bbox_inches='tight', dpi=150)
        fig_mpl.savefig(svg_path, bbox_inches='tight')
        plt.close(fig_mpl)

        csv_out = full_df.rename(columns={phase_col: 'Phase', tech_col: 'Technology'})
        csv_out.to_csv(self.outdir + f"Decision_tracking/F_decided_realized_up_to_heatmap_share_{method_label}{tech_suffix}.csv", index=False)

        return csv_out


    def _decision_heatmap_core(self, decision_tracking, selected_techs,
                               sample_col, tech_col, phase_col,
                               zero_display_tolerance, min_capacity_gw,
                               colorscale_plotly, cmap_colors,
                               file_suffix, plot, show_legend_export):
        """Noyau commun aux heatmaps de décision par secteur."""
        from matplotlib.colors import LinearSegmentedColormap

        phase_order = sorted(decision_tracking[phase_col].dropna().unique().tolist(), key=self._phase_sort_key)
        phase_order_map = {phase: i for i, phase in enumerate(phase_order)}

        grouped = (
            decision_tracking
            .groupby([sample_col, tech_col, phase_col], as_index=False)['Decision_GW']
            .sum()
        )
        grouped['__phase_order'] = grouped[phase_col].map(phase_order_map)
        grouped.sort_values(by=[sample_col, tech_col, '__phase_order'], inplace=True)

        grouped['has_decision'] = (grouped['Decision_GW'] > 0.001).astype(float)
        aggregated = grouped.groupby([tech_col, phase_col], as_index=False)['has_decision'].mean()
        aggregated.rename(columns={'has_decision': 'Share_pct'}, inplace=True)

        tech_order = [t for t in selected_techs if t in set(aggregated[tech_col])]
        if not tech_order:
            raise ValueError(f"Aucune technologie avec des décisions à tracer ({file_suffix}).")

        full_index = pd.MultiIndex.from_product([tech_order, phase_order], names=[tech_col, phase_col])
        full_df = pd.DataFrame(index=full_index).reset_index()
        full_df = full_df.merge(aggregated, on=[tech_col, phase_col], how='left')
        full_df['Share_pct'] = full_df['Share_pct'].fillna(0)
        full_df['Share_pct'] = full_df['Share_pct'].where(full_df['Share_pct'] >= float(zero_display_tolerance), 0.0)

        matrix = (
            full_df.pivot(index=tech_col, columns=phase_col, values='Share_pct')
            .reindex(index=tech_order, columns=phase_order).fillna(0)
        )
        matrix = matrix.mask(matrix < float(zero_display_tolerance), 0.0)

        tech_stats = (
            grouped.groupby([sample_col, tech_col], as_index=False)['Decision_GW'].sum()
            .groupby(tech_col)['Decision_GW'].agg(['min', 'median', 'max'])
            .reindex(matrix.index).fillna(0.0)
        )
        active_techs = [t for t in matrix.index if tech_stats.loc[t, 'max'] >= min_capacity_gw]
        matrix     = matrix.loc[active_techs]
        tech_stats = tech_stats.loc[active_techs]
        tech_order = active_techs

        meaning = self.dict_meaning()
        gray = 'rgb(90,90,90)'

        def _fmt_gw(v):
            s = f"{v:.1f}"
            return s[:-2] if s.endswith('.0') else s

        def _fmt_tech(tech):
            raw = meaning.get(tech, tech)
            parts = raw.split('_')
            return parts[0] + (' ' + ' '.join(p.lower() for p in parts[1:]) if len(parts) > 1 else '')

        tech_labels = [
            f"{_fmt_tech(t)} : {_fmt_gw(tech_stats.loc[t, 'median'])} [{_fmt_gw(tech_stats.loc[t, 'min'])}, {_fmt_gw(tech_stats.loc[t, 'max'])}]"
            for t in matrix.index
        ]

        # Médiane de capacité décidée — uniquement sur les scénarios qui décident
        median_phase = (
            grouped[grouped['Decision_GW'] > 0.001]
            .groupby([tech_col, phase_col])['Decision_GW']
            .median()
            .reset_index()
            .pivot(index=tech_col, columns=phase_col, values='Decision_GW')
            .reindex(index=active_techs, columns=phase_order)
            .fillna(0.0)
        )
        # Cellules où la médiane est trop petite → on les traite comme "pas de décision"
        small_cells = (median_phase < min_capacity_gw) | (matrix == 0)
        matrix = matrix.where(~small_cells, 0.0)
        text_matrix = median_phase.where(~small_cells).applymap(
            lambda v: _fmt_gw(v) if pd.notna(v) else ""
        )

        def _phase_start_year(p):
            parts = str(p).replace('_', '-').split('-')
            nums = [x for x in parts if x.isdigit() and len(x) == 4]
            return nums[0] if nums else str(p)

        phase_end_years  = [_phase_start_year(p) for p in phase_order]
        _last_year       = phase_end_years[-1] if phase_end_years else None
        x_tickvals_shown = [y for y in phase_end_years if (y.isdigit() and int(y) % 5 == 0) or y == _last_year]

        cell_px = 46
        n_cols  = max(1, len(phase_end_years))
        n_rows  = max(1, len(tech_order))
        lm, rm, tm, bm = 260, 110, 40, 55
        pw = max(900,  lm + rm + cell_px * n_cols)
        ph = max(300, tm + bm + cell_px * n_rows)

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values, x=phase_end_years, y=tech_labels,
            text=text_matrix.values,
            texttemplate='%{text}',
            textfont=dict(size=9, color='black'),
            zmin=0, zmax=1, colorscale=colorscale_plotly,
            colorbar=dict(
                title=dict(text='% of scenarios', font=dict(size=13, color=gray)),
                tickmode='array', tickvals=[0, 0.5, 1],
                ticktext=['0%', '50%', '100%'], tickfont=dict(size=12, color=gray),
                thickness=14, len=0.8,
            ),
            hovertemplate='%{y}<br>Year: %{x}<br>Scenarios: %{z:.0%}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text=''), template='simple_white',
            width=pw, height=ph,
            margin=dict(l=lm, r=rm, t=tm, b=bm),
            font=dict(family='Rawline', color=gray, size=12),
            xaxis=dict(automargin=False, tickmode='array',
                       tickvals=x_tickvals_shown, ticktext=x_tickvals_shown,
                       tickangle=0, tickfont=dict(size=20, color=gray),
                       tickcolor=gray, ticks='outside', ticklen=4),
            yaxis=dict(automargin=False, categoryorder='array',
                       categoryarray=tech_labels,
                       tickfont=dict(size=20, color='black'), ticks=''),
        )

        if plot:
            pio.show(fig)

        out_dir = self.outdir + "Decision_tracking/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(out_dir + f"_Raw/F_decided_{file_suffix}.html")

        cmap = LinearSegmentedColormap.from_list('_cmap', cmap_colors)
        mat  = matrix.values
        n_r, n_c = mat.shape
        fig_w = max(10, n_c * 0.42 + 4.5)
        fig_h = max(3,  n_r * 0.55 + 1.5)
        fig_mpl, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

        _shown_set     = set(x_tickvals_shown)
        _x_idx         = [i for i, y in enumerate(phase_end_years) if y in _shown_set]
        ax.set_xticks(_x_idx)
        ax.set_xticklabels([phase_end_years[i] for i in _x_idx], fontsize=20, color='#5a5a5a', rotation=0)
        ax.set_yticks(range(n_r))
        ax.set_yticklabels(tech_labels, fontsize=20, color='black')
        ax.tick_params(length=0)

        # Annotations texte (médiane GW) — blanc si cellule foncée
        text_vals = text_matrix.values
        for row in range(n_r):
            for col in range(n_c):
                txt = text_vals[row, col]
                if txt:
                    txt_color = 'white' if mat[row, col] > 0.5 else 'black'
                    ax.text(col, row, txt, ha='center', va='center',
                            fontsize=12, color=txt_color, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_visible(False)

        if show_legend_export:
            cbar = fig_mpl.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['0%', '50%', '100%'])
            cbar.ax.tick_params(labelsize=9, colors='#5a5a5a')
            cbar.outline.set_visible(False)
            cbar.set_label('% of scenarios', fontsize=10, color='#5a5a5a')

        fig_mpl.tight_layout()
        fig_mpl.savefig(out_dir + f"F_decided_{file_suffix}.pdf", bbox_inches='tight', dpi=150)
        fig_mpl.savefig(out_dir + f"F_decided_{file_suffix}.svg", bbox_inches='tight')
        plt.close(fig_mpl)

        csv_out = full_df.rename(columns={phase_col: 'Phase', tech_col: 'Technology'})
        csv_out.to_csv(out_dir + f"F_decided_{file_suffix}_share.csv", index=False)
        return csv_out

    # ── Couleurs par défaut ───────────────────────────────────────────────────
    _CS_HEAT = [
        [0.00, 'rgb(255,247,236)'], [0.15, 'rgb(254,210,166)'],
        [0.35, 'rgb(253,141,60)'],  [0.60, 'rgb(217,71,1)'],
        [0.80, 'rgb(166,54,3)'],    [1.00, 'rgb(127,39,4)'],
    ]
    _CM_HEAT = ['#fff7ec', '#fdd0a2', '#fd8d3c', '#d94801', '#7f2704']

    _CS_MOB = [
        [0.00, 'rgb(255,255,204)'], [0.15, 'rgb(255,237,160)'],
        [0.35, 'rgb(254,196,79)'],  [0.60, 'rgb(254,153,41)'],
        [0.80, 'rgb(204,76,2)'],    [1.00, 'rgb(102,37,6)'],
    ]
    _CM_MOB = ['#ffffe0', '#ffeda0', '#feb24c', '#fc4e2a', '#800026']

    _CS_FRT = [
        [0.00, 'rgb(247,252,245)'], [0.15, 'rgb(199,233,192)'],
        [0.35, 'rgb(116,196,118)'], [0.60, 'rgb(49,163,84)'],
        [0.80, 'rgb(0,109,44)'],    [1.00, 'rgb(0,68,27)'],
    ]
    _CM_FRT = ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45', '#00441b']

    def graph_decision_timing_heatmap_heat(self, ampl_uq_collector=None, plot=True,
                                           show_legend_export=True,
                                           zero_display_tolerance=0.005,
                                           layer='all',
                                           exclude_technologies=None,
                                           include_ccgt=False,
                                           min_capacity_gw=0.1):
        """
        Heatmap du timing des décisions pour les technologies de chaleur.

        Identique à graph_decision_timing_heatmap mais filtrée sur les
        technologies de chaleur (HEAT_HIGH_T, HEAT_LOW_T_DHN, HEAT_LOW_T_DECEN).

        Parameters
        ----------
        layer : str
            'all'           → toutes les technologies de chaleur,
            'HEAT_HIGH_T'   → haute température uniquement,
            'HEAT_LOW_T_DHN'   → basse temp. réseau,
            'HEAT_LOW_T_DECEN' → basse temp. décentralisé.
        exclude_technologies : list | None
            Technologies supplémentaires à exclure.
        include_ccgt : bool
            Si False (défaut), retire les technologies contenant 'CCGT'.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy().reset_index()

        if 'F_decided_realized_up_to' not in decision_tracking.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col  = 'Phases'      if 'Phases'      in decision_tracking.columns else decision_tracking.columns[0]
        tech_col   = 'Technologies' if 'Technologies' in decision_tracking.columns else decision_tracking.columns[1]
        sample_col = 'Sample'
        if sample_col not in decision_tracking.columns:
            decision_tracking[sample_col] = 0
        n_scenarios = int(decision_tracking[sample_col].nunique())

        decision_tracking = decision_tracking[[sample_col, phase_col, tech_col, 'F_decided_realized_up_to']].copy()
        decision_tracking.rename(columns={'F_decided_realized_up_to': 'Decision_GW'}, inplace=True)

        all_dt = set(decision_tracking[tech_col].unique())
        _high_t = [t for t in all_dt if t.startswith('IND_')]
        _dhn    = [t for t in all_dt if t.startswith('DHN_')]
        _decen  = [t for t in all_dt if t.startswith('DEC_')]
        _lmap   = {'HEAT_HIGH_T': _high_t, 'HEAT_LOW_T_DHN': _dhn,
                   'HEAT_LOW_T_DECEN': _decen, 'all': _high_t + _dhn + _decen}
        techs = list(dict.fromkeys(_lmap.get(str(layer), _lmap['all'])))

        excl = {'DEC_DIRECT_ELEC', 'DHN_BOILER_OIL'} | (set(exclude_technologies) if exclude_technologies else set())
        if not include_ccgt:
            excl |= {t for t in techs if 'CCGT' in t.upper()}
        techs = [t for t in techs if t not in excl]

        dt = decision_tracking.loc[decision_tracking[tech_col].isin(techs)].copy()
        if dt.empty:
            raise ValueError(f"Aucune donnée de décision trouvée pour les technologies de chaleur ({layer}).")

        suffix = layer if layer != 'all' else 'heat_all'
        return self._decision_heatmap_core(
            dt, techs, sample_col, tech_col, phase_col,
            zero_display_tolerance, min_capacity_gw,
            self._CS_HEAT, self._CM_HEAT, f"heat_heatmap_{suffix}", plot, show_legend_export
        )


    def graph_decision_timing_heatmap_mobility(self, ampl_uq_collector=None, plot=True,
                                               show_legend_export=True,
                                               zero_display_tolerance=0.005,
                                               layer='all',
                                               exclude_technologies=None,
                                               min_capacity_gw=0.1):
        """Heatmap du timing des décisions pour les technologies de mobilité.

        layer : 'MOB_PRIVATE' | 'MOB_PUBLIC' | 'all'
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector
        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in dt.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col  = 'Phases'       if 'Phases'       in dt.columns else dt.columns[0]
        tech_col   = 'Technologies' if 'Technologies'  in dt.columns else dt.columns[1]
        sample_col = 'Sample'
        if sample_col not in dt.columns:
            dt[sample_col] = 0
        dt = dt[[sample_col, phase_col, tech_col, 'F_decided_realized_up_to']].copy()
        dt.rename(columns={'F_decided_realized_up_to': 'Decision_GW'}, inplace=True)

        all_dt   = set(dt[tech_col].unique())
        _private = [t for t in all_dt if t.startswith('CAR_')]
        _public  = [t for t in all_dt if t.startswith(('BUS_', 'TRAMWAY_')) or t == 'TRAIN_PUB']
        _lmap    = {'MOB_PRIVATE': _private, 'MOB_PUBLIC': _public, 'all': _private + _public}
        techs = list(dict.fromkeys(_lmap.get(str(layer), _lmap['all'])))

        excl = set(exclude_technologies) if exclude_technologies else set()
        techs = [t for t in techs if t not in excl]

        dt = dt.loc[dt[tech_col].isin(techs)].copy()
        if dt.empty:
            raise ValueError(f"Aucune donnée de décision pour les technologies de mobilité ({layer}).")

        suffix = layer if layer != 'all' else 'mobility_all'
        return self._decision_heatmap_core(
            dt, techs, sample_col, tech_col, phase_col,
            zero_display_tolerance, min_capacity_gw,
            self._CS_MOB, self._CM_MOB, f"mobility_heatmap_{suffix}", plot, show_legend_export
        )


    def graph_decision_timing_heatmap_freight(self, ampl_uq_collector=None, plot=True,
                                              show_legend_export=True,
                                              zero_display_tolerance=0.005,
                                              exclude_technologies=None,
                                              min_capacity_gw=0.1):
        """Heatmap du timing des décisions pour les technologies de fret."""
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector
        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        if 'F_decided_realized_up_to' not in dt.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col  = 'Phases'       if 'Phases'       in dt.columns else dt.columns[0]
        tech_col   = 'Technologies' if 'Technologies'  in dt.columns else dt.columns[1]
        sample_col = 'Sample'
        if sample_col not in dt.columns:
            dt[sample_col] = 0
        dt = dt[[sample_col, phase_col, tech_col, 'F_decided_realized_up_to']].copy()
        dt.rename(columns={'F_decided_realized_up_to': 'Decision_GW'}, inplace=True)

        all_dt = set(dt[tech_col].unique())
        techs  = [t for t in all_dt if t.startswith(('TRUCK_', 'BOAT_FREIGHT_')) or t == 'TRAIN_FREIGHT']
        techs  = list(dict.fromkeys(techs))

        excl  = set(exclude_technologies) if exclude_technologies else set()
        techs = [t for t in techs if t not in excl]

        dt = dt.loc[dt[tech_col].isin(techs)].copy()
        if dt.empty:
            raise ValueError("Aucune donnée de décision pour les technologies de fret.")

        return self._decision_heatmap_core(
            dt, techs, sample_col, tech_col, phase_col,
            zero_display_tolerance, min_capacity_gw,
            self._CS_FRT, self._CM_FRT, "freight_heatmap", plot, show_legend_export
        )


    def graph_decision_timing_heatmap_all_sectors(self, ampl_uq_collector=None, plot=True,
                                                  show_legend_export=True,
                                                  zero_display_tolerance=0.005,
                                                  min_capacity_gw=0.1,
                                                  sectors=None):
        """
        Génère une heatmap de timing de décision pour chaque secteur défini dans dict_color.

        Appelle _decision_heatmap_core pour chaque secteur avec les technologies
        correspondantes extraites de dict_color(sector).keys().

        Parameters
        ----------
        sectors : list | None
            Secteurs à traiter. Si None, tous les secteurs sont traités.
            Options valides : 'Electricity', 'Heat_high_T', 'Heat_low_T_DHN',
            'Heat_low_T_Decen', 'Mobility', 'Freight', 'Ammonia', 'Methanol',
            'HVC', 'Conversion', 'Storage'.

        Returns
        -------
        dict[str, pd.DataFrame]
            Un DataFrame par secteur (clé = nom du secteur), contenant les
            colonnes Phase / Technology / Share_pct.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Decision_tracking' not in ampl_uq_collector:
            raise ValueError("'Decision_tracking' introuvable dans le collecteur UQ.")

        decision_tracking = ampl_uq_collector['Decision_tracking'].copy().reset_index()

        if 'F_decided_realized_up_to' not in decision_tracking.columns:
            raise ValueError("Colonne 'F_decided_realized_up_to' introuvable dans 'Decision_tracking'.")

        phase_col  = 'Phases'       if 'Phases'       in decision_tracking.columns else decision_tracking.columns[0]
        tech_col   = 'Technologies' if 'Technologies' in decision_tracking.columns else decision_tracking.columns[1]
        sample_col = 'Sample'
        if sample_col not in decision_tracking.columns:
            decision_tracking[sample_col] = 0

        decision_tracking = decision_tracking[[sample_col, phase_col, tech_col, 'F_decided_realized_up_to']].copy()
        decision_tracking.rename(columns={'F_decided_realized_up_to': 'Decision_GW'}, inplace=True)

        _CS_ELEC = [
            [0.00, 'rgb(247,251,255)'], [0.15, 'rgb(198,219,239)'],
            [0.35, 'rgb(107,174,214)'], [0.60, 'rgb(49,130,189)'],
            [0.80, 'rgb(22,96,162)'],   [1.00, 'rgb(8,48,107)'],
        ]
        _CM_ELEC = ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b']

        _CS_PURP = [
            [0.00, 'rgb(252,251,253)'], [0.15, 'rgb(218,218,235)'],
            [0.35, 'rgb(158,154,200)'], [0.60, 'rgb(117,107,177)'],
            [0.80, 'rgb(84,39,143)'],   [1.00, 'rgb(63,0,125)'],
        ]
        _CM_PURP = ['#fcfbfd', '#dadaeb', '#9e9ac8', '#756bb1', '#3f007d']

        _CS_GRAY = [
            [0.00, 'rgb(255,255,255)'], [0.15, 'rgb(217,217,217)'],
            [0.35, 'rgb(150,150,150)'], [0.60, 'rgb(99,99,99)'],
            [0.80, 'rgb(50,50,50)'],    [1.00, 'rgb(0,0,0)'],
        ]
        _CM_GRAY = ['#ffffff', '#d9d9d9', '#969696', '#636363', '#252525']

        heat_low_t_all = list(self.dict_color('Heat_low_T').keys())
        sector_specs = [
            ('Electricity',
             [t for t in self.dict_color('Electricity').keys() if t != 'ELECTRICITY'],
             _CS_ELEC, _CM_ELEC, 'electricity'),
            ('Heat_high_T',
             list(self.dict_color('Heat_high_T').keys()),
             self._CS_HEAT, self._CM_HEAT, 'heat_high_t'),
            ('Heat_low_T_DHN',
             [t for t in heat_low_t_all if t.startswith('DHN_')],
             self._CS_HEAT, self._CM_HEAT, 'heat_low_t_dhn'),
            ('Heat_low_T_Decen',
             [t for t in heat_low_t_all if t.startswith('DEC_')],
             self._CS_HEAT, self._CM_HEAT, 'heat_low_t_decen'),
            ('Mobility',
             list(self.dict_color('Mobility').keys()),
             self._CS_MOB, self._CM_MOB, 'mobility'),
            ('Freight',
             list(self.dict_color('Freight').keys()),
             self._CS_FRT, self._CM_FRT, 'freight'),
            ('Ammonia',
             list(self.dict_color('Ammonia').keys()),
             _CS_PURP, _CM_PURP, 'ammonia'),
            ('Methanol',
             list(self.dict_color('Methanol').keys()),
             _CS_PURP, _CM_PURP, 'methanol'),
            ('HVC',
             list(self.dict_color('HVC').keys()),
             _CS_PURP, _CM_PURP, 'hvc'),
            ('Conversion',
             list(self.dict_color('Conversion').keys()),
             _CS_GRAY, _CM_GRAY, 'conversion'),
            ('Storage',
             list(self.dict_color('Storage').keys()),
             _CS_GRAY, _CM_GRAY, 'storage'),
        ]

        if sectors is not None:
            sector_specs = [s for s in sector_specs if s[0] in sectors]

        results = {}
        for sector_name, techs, cs, cm, suffix in sector_specs:
            dt = decision_tracking.loc[decision_tracking[tech_col].isin(techs)].copy()
            if dt.empty:
                continue
            try:
                results[sector_name] = self._decision_heatmap_core(
                    dt, techs, sample_col, tech_col, phase_col,
                    zero_display_tolerance, min_capacity_gw,
                    cs, cm, suffix, plot, show_legend_export
                )
            except ValueError:
                pass

        return results


    def graph_installed_capacity_variation(self, ampl_uq_collector=None, plot=True,
                                            year=2050, technologies=None,
                                            min_capacity_gw=0.1):
        """
        Boxplot de la capacité installée (F) par technologie à une année donnée,
        sur tous les samples UQ.

        Parameters
        ----------
        year : int
            Année cible (ex: 2050).
        technologies : list | None
            Liste de technologies à afficher. Si None, toutes les technologies
            significatives sont affichées.
        min_capacity_gw : float
            Seuil (GW) sous lequel une technologie est ignorée (médiane < seuil).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Assets' not in ampl_uq_collector:
            raise ValueError("'Assets' introuvable dans le collecteur UQ.")

        assets = ampl_uq_collector['Assets'].copy().reset_index()

        year_col = next((c for c in assets.columns if c in ('Years', 'Year', 'Phases')), None)
        tech_col = next((c for c in assets.columns if 'Tech' in c or c == 'Technologies'), None)
        if year_col is None or tech_col is None:
            raise ValueError("Colonnes 'Technologies' ou 'Years' introuvables dans Assets.")

        assets['_year'] = assets[year_col].astype(str).str.extract(r'(\d{4})').astype(int)
        df = assets[assets['_year'] == int(year)].copy()

        if df.empty:
            print(f"graph_installed_capacity_variation: aucune donnée pour l'année {year}.")
            return pd.DataFrame()

        df['F'] = pd.to_numeric(df['F'], errors='coerce').fillna(0)

        if technologies is not None:
            df = df[df[tech_col].isin(technologies)]

        sig = df.groupby(tech_col)['F'].median()
        df = df[df[tech_col].isin(sig[sig >= min_capacity_gw].index)]

        if df.empty:
            print("graph_installed_capacity_variation: aucune technologie significative.")
            return pd.DataFrame()

        order = df.groupby(tech_col)['F'].median().sort_values(ascending=False).index.tolist()

        if not plot:
            return df

        fig = px.box(
            df, x=tech_col, y='F',
            color=tech_col,
            title=f'Installed capacity in {year} — variation across scenarios [GW]',
            color_discrete_map=self.color_dict_full,
            points='outliers', notched=False,
            category_orders={tech_col: order},
        )
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        pio.show(fig)

        outdir = self.outdir + f"InstalledCapacity/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/installed_capacity_{year}_raw.html")

        yvals = [0, round(float(df['F'].max()), 1)]
        title_str = f"<b>Installed capacity in {year} — variation across scenarios</b><br>[GW]"
        self.custom_fig(fig, title_str, yvals, type_graph='bar')
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        fig.write_image(outdir + f"installed_capacity_{year}.pdf", width=1600, height=600)
        plt.close()

        return df

    def graph_layer_balance_variation(self, layer='ELECTRICITY', ampl_uq_collector=None,
                                       plot=True, year=2050, min_twh=1.0):
        """
        Boxplot de la production/consommation par technologie pour un layer donné
        à une année cible, sur tous les samples UQ.

        Parameters
        ----------
        layer : str
            Layer énergétique à analyser (ex: 'ELECTRICITY', 'H2', 'HEAT_LOW_T_DHN').
        year : int
            Année cible.
        min_twh : float
            Seuil absolu (TWh) sous lequel un élément est ignoré (médiane < seuil).
            Seules les valeurs positives (production) sont conservées.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Year_balance' not in ampl_uq_collector:
            raise ValueError("'Year_balance' introuvable dans le collecteur UQ.")

        yb = ampl_uq_collector['Year_balance'].copy().reset_index()

        if layer not in yb.columns:
            raise ValueError(f"Layer '{layer}' introuvable dans Year_balance. "
                             f"Colonnes disponibles : {list(yb.columns)}")

        year_col = next((c for c in yb.columns if c in ('Years', 'Year')), None)
        elem_col = next((c for c in yb.columns if c in ('Elements', 'Element', 'Technologies')), None)
        if year_col is None or elem_col is None:
            raise ValueError("Colonnes 'Elements' ou 'Years' introuvables dans Year_balance.")

        yb['_year'] = yb[year_col].astype(str).str.extract(r'(\d{4})').astype(int)
        df = yb[yb['_year'] == int(year)][['Sample', elem_col, layer]].copy()
        df[layer] = pd.to_numeric(df[layer], errors='coerce').fillna(0)

        # Garder uniquement les producteurs (valeurs positives)
        df = df[df[layer] > 0]

        if df.empty:
            print(f"graph_layer_balance_variation: aucune donnée pour {layer} en {year}.")
            return pd.DataFrame()

        sig = df.groupby(elem_col)[layer].median()
        df = df[df[elem_col].isin(sig[sig >= min_twh].index)]

        if df.empty:
            print(f"graph_layer_balance_variation: aucun élément significatif (seuil {min_twh} TWh).")
            return pd.DataFrame()

        order = df.groupby(elem_col)[layer].median().sort_values(ascending=False).index.tolist()

        if not plot:
            return df

        fig = px.box(
            df, x=elem_col, y=layer,
            color=elem_col,
            title=f'{layer} production in {year} — variation across scenarios [TWh]',
            color_discrete_map=self.color_dict_full,
            points='outliers', notched=False,
            category_orders={elem_col: order},
        )
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        pio.show(fig)

        outdir = self.outdir + f"LayerBalance/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"_Raw/{layer}_{year}_balance_raw.html")

        yvals = [0, round(float(df[layer].max()), 1)]
        title_str = f"<b>{layer} production in {year} — variation across scenarios</b><br>[TWh]"
        self.custom_fig(fig, title_str, yvals, type_graph='bar')
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        fig.write_image(outdir + f"{layer}_{year}_balance.pdf", width=1600, height=600)
        plt.close()

        return df

    def graph_capacity_commissioning_correlation(self, ampl_uq_collector=None, plot=True,
                                                   year=2050, min_capacity_gw=0.1,
                                                   method='spearman', technologies=None):
        """
        Heatmap de corrélation entre les paramètres de commissioning time (axe X)
        et la capacité installée (F) de chaque technologie en `year` (axe Y).

        Parameters
        ----------
        year : int
            Année cible pour la capacité installée.
        min_capacity_gw : float
            Seuil médian (GW) sous lequel une technologie est ignorée.
        method : str
            'spearman' (défaut, robuste aux non-linéarités) ou 'pearson'.
        technologies : list | None
            Filtre optionnel sur les technologies affichées.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        if 'Assets' not in ampl_uq_collector:
            raise ValueError("'Assets' introuvable dans le collecteur UQ.")
        if 'Samples' not in ampl_uq_collector:
            raise ValueError("'Samples' introuvable dans le collecteur UQ.")

        # ── Commissioning time params (colonnes cp_*) ─────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index()
        ct_cols = [c for c in samples.columns
                   if isinstance(c, str) and c.startswith('cp_')]
        if not ct_cols:
            raise ValueError("Aucune colonne cp_* de commissioning time trouvée dans Samples.")

        # Technologies dont le commissioning time est incertain = noms après 'cp_'
        ct_techs = [c[len('cp_'):] for c in ct_cols]

        sample_col = 'Sample' if 'Sample' in samples.columns else samples.columns[0]
        df_ct = samples[[sample_col] + ct_cols].copy()
        df_ct[sample_col] = df_ct[sample_col].astype(str)

        # ── Capacité installée à l'année cible ────────────────────────────────
        assets = ampl_uq_collector['Assets'].copy().reset_index()
        year_col = next((c for c in assets.columns if c in ('Years', 'Year', 'Phases')), None)
        tech_col = next((c for c in assets.columns if 'Tech' in c or c == 'Technologies'), None)
        s_col    = next((c for c in assets.columns if c == 'Sample'), None)
        if year_col is None or tech_col is None or s_col is None:
            raise ValueError("Colonnes manquantes dans Assets (Technologies, Years, Sample).")

        assets['_year'] = assets[year_col].astype(str).str.extract(r'(\d{4})').astype(int)
        df_f = assets[assets['_year'] == int(year)][[s_col, tech_col, 'F']].copy()
        df_f['F'] = pd.to_numeric(df_f['F'], errors='coerce').fillna(0)
        df_f[s_col] = df_f[s_col].astype(str)

        # Restreindre aux technologies avec un commissioning time incertain
        filter_techs = technologies if technologies is not None else ct_techs
        df_f = df_f[df_f[tech_col].isin(filter_techs)]

        # Filtrer techs insignifiantes (médiane trop faible)
        sig = df_f.groupby(tech_col)['F'].median()
        df_f = df_f[df_f[tech_col].isin(sig[sig >= min_capacity_gw].index)]

        # Filtrer techs quasi-constantes (CV < 0.1%) — évite corrélations spurieuses
        grp  = df_f.groupby(tech_col)['F']
        cv   = grp.std(ddof=0) / grp.mean().replace(0, np.nan)
        df_f = df_f[df_f[tech_col].isin(cv[cv.fillna(0) > 0.001].index)]

        if df_f.empty:
            print("graph_capacity_commissioning_correlation: aucune technologie significative.")
            return pd.DataFrame()

        # Pivot : lignes = samples, colonnes = technologies
        df_pivot = df_f.pivot_table(index=s_col, columns=tech_col, values='F', aggfunc='mean')

        # Merge avec commissioning times sur l'index Sample
        df_merged = df_ct.set_index(sample_col).join(df_pivot, how='inner')
        if df_merged.empty:
            raise ValueError("Aucun sample commun entre Samples et Assets.")

        # ── Matrice de corrélation ─────────────────────────────────────────────
        tech_names  = list(df_pivot.columns)
        corr_matrix = pd.DataFrame(index=tech_names, columns=ct_cols, dtype=float)

        for tech in tech_names:
            for ct in ct_cols:
                s = df_merged[[ct, tech]].dropna()
                if len(s) < 3:
                    corr_matrix.loc[tech, ct] = float('nan')
                    continue
                if method == 'spearman':
                    r = s[ct].rank().corr(s[tech].rank())
                else:
                    r = s[ct].corr(s[tech])
                corr_matrix.loc[tech, ct] = round(float(r), 3)

        # Trier les technologies par corrélation max absolue (les plus sensibles en haut)
        corr_matrix['_max'] = corr_matrix[ct_cols].abs().max(axis=1)
        corr_matrix.sort_values('_max', ascending=False, inplace=True)
        corr_matrix.drop(columns='_max', inplace=True)

        # Labels lisibles pour les colonnes (commissioning time)
        ct_labels = [self.uncert_param_meaning.get(c, c) for c in ct_cols]
        tech_labels = list(corr_matrix.index)

        if not plot:
            return corr_matrix

        outdir = self.outdir + "CapacityCorrelation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        corr_matrix.to_csv(outdir + f"corr_capacity_ct_{year}.csv")

        title = f"Correlation — commissioning time vs installed capacity {year} ({method})"
        self._export_corr_heatmap(
            row_matrix=corr_matrix,
            x_labels=ct_labels,
            y_labels=tech_labels,
            title=title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem=f"corr_capacity_ct_{year}",
            decimals=2,
            text_threshold=0.1,
        )
        return corr_matrix

    def summary_capacity(self, year=2040, ampl_uq_collector=None, min_median_gw=0.1,
                         force_include=None):
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector
        ct_techs = [c[len('cp_'):] for c in ampl_uq_collector['Samples'].columns
                    if isinstance(c, str) and c.startswith('cp_')]
        if force_include:
            ct_techs = list(set(ct_techs) | set(force_include))
        assets = ampl_uq_collector['Assets'].copy().reset_index()
        year_col = next((c for c in assets.columns if c in ('Years', 'Year', 'Phases')), None)
        tech_col = next((c for c in assets.columns if 'Tech' in c or c == 'Technologies'), None)
        assets['_year'] = assets[year_col].astype(str).str.extract(r'(\d{4})').astype(int)
        df = assets[(assets['_year'] == int(year)) & (assets[tech_col].isin(ct_techs))].copy()
        df['F'] = pd.to_numeric(df['F'], errors='coerce')
        summary = (df.groupby(tech_col)['F']
                     .agg(['min', 'median', 'max'])
                     .round(2)
                     .sort_values('median', ascending=False))
        forced = set(force_include or [])
        summary = summary[(summary['median'] >= min_median_gw) | summary.index.isin(forced)]
        print(f"\n── Installed capacity in {year} [GW] — commissioning time technologies ──")
        print(summary.to_string())
        return summary

    def summary_commissioning_time(self, year='all', ampl_uq_collector=None,
                                    min_decided_gw=0.1, technologies=None):
        """
        Pour chaque technologie avec un cp_*, affiche les stats du commissioning time
        sur les scénarios où la capacité décidée dépasse min_decided_gw.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        samples['Sample'] = samples['Sample'].astype(str)

        ct_cols  = [c for c in samples.columns if isinstance(c, str) and c.startswith('cp_')]
        ct_techs = [c[len('cp_'):] for c in ct_cols]
        if technologies is not None:
            ct_cols  = [c for c, t in zip(ct_cols, ct_techs) if t in technologies]
            ct_techs = [t for t in ct_techs if t in technologies]

        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        decided_col = next((c for c in ('F_decided_realized_up_to', 'F_decided_up_to')
                            if c in dt.columns), None)
        if decided_col is None:
            raise ValueError("Colonne de capacité décidée introuvable dans Decision_tracking.")

        dt['Sample'] = pd.to_numeric(dt['Sample'], errors='coerce').astype(str)
        dt[decided_col] = pd.to_numeric(dt[decided_col], errors='coerce')

        if year != 'all':
            phase_col = next((c for c in dt.columns if c in ('Phases', 'Phase')), None)
            if phase_col:
                dt['_phase_start'] = dt[phase_col].astype(str).str.extract(r'(\d{4})').astype(float)
                dt = dt[dt['_phase_start'] == float(year)]

        cap = (dt.dropna(subset=['Sample', 'Technologies', decided_col])
                 .groupby(['Sample', 'Technologies'], as_index=False)[decided_col].sum())

        records = []
        for cp_col, tech in zip(ct_cols, ct_techs):
            cap_tech = cap[cap['Technologies'] == tech]
            active_samples = cap_tech.loc[cap_tech[decided_col] > min_decided_gw, 'Sample']
            if active_samples.empty:
                continue
            ct_vals = np.exp(pd.to_numeric(
                samples.loc[samples['Sample'].isin(active_samples), cp_col], errors='coerce'
            ).dropna())
            if ct_vals.empty:
                continue
            records.append({
                'Technology': tech,
                'n_scenarios': int(len(active_samples)),
                'ct_min':    round(float(ct_vals.min()), 2),
                'ct_median': round(float(ct_vals.median()), 2),
                'ct_max':    round(float(ct_vals.max()), 2),
            })

        if not records:
            print("Aucune technologie avec capacité décidée > seuil.")
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index('Technology').sort_values('ct_median', ascending=False)
        year_str = str(year)
        print(f"\n── Commissioning time [years] for scenarios with decided capacity > {min_decided_gw} GW"
              f" (year={year_str}) ──")
        print(result.to_string())
        return result

    def scenarios_capacity_commissioning(self, technology, year='all',
                                          ampl_uq_collector=None):
        """
        Pour chaque scénario, retourne la capacité décidée et le commissioning time
        d'une technologie donnée.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        samples['Sample'] = samples['Sample'].astype(str)

        cp_col = f'cp_{technology}'
        if cp_col not in samples.columns:
            raise ValueError(f"'{cp_col}' introuvable dans Samples.")

        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        decided_col = next((c for c in ('F_decided_realized_up_to', 'F_decided_up_to')
                            if c in dt.columns), None)
        if decided_col is None:
            raise ValueError("Colonne de capacité décidée introuvable dans Decision_tracking.")

        dt['Sample'] = pd.to_numeric(dt['Sample'], errors='coerce').astype(str)
        dt[decided_col] = pd.to_numeric(dt[decided_col], errors='coerce')

        if year != 'all':
            phase_col = next((c for c in dt.columns if c in ('Phases', 'Phase')), None)
            if phase_col:
                dt['_phase_start'] = dt[phase_col].astype(str).str.extract(r'(\d{4})').astype(float)
                dt = dt[dt['_phase_start'] == float(year)]

        cap = (dt[dt['Technologies'] == technology]
                 .dropna(subset=['Sample', decided_col])
                 .groupby('Sample', as_index=False)[decided_col].sum()
                 .rename(columns={decided_col: 'decided_capacity_GW'}))

        ct = samples[['Sample', cp_col]].copy()
        ct[cp_col] = np.exp(pd.to_numeric(ct[cp_col], errors='coerce'))
        ct.rename(columns={cp_col: 'commissioning_time_years'}, inplace=True)

        result = cap.merge(ct, on='Sample', how='inner').sort_values('Sample').round(2)
        result = result[result['decided_capacity_GW'] != 0.00]

        print(f"\n── {technology} — decided capacity [GW] & commissioning time [years]"
              f" per scenario (year={year}) ──")
        print(result.to_string(index=False))
        return result

    def graph_capacity_vs_commissioning(self, technology, ampl_uq_collector=None, plot=True):
        """
        Scatter plot : commissioning time (x) vs capacité décidée totale (y) par scénario.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        samples['Sample'] = samples['Sample'].astype(str)

        cp_col = f'cp_{technology}'
        if cp_col not in samples.columns:
            raise ValueError(f"'{cp_col}' introuvable dans Samples.")

        ct = samples[['Sample', cp_col]].copy()
        ct['commissioning_time'] = np.exp(pd.to_numeric(ct[cp_col], errors='coerce'))
        ct = ct.drop(columns=[cp_col])

        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        decided_col = next((c for c in ('F_decided_realized_up_to', 'F_decided_up_to')
                            if c in dt.columns), None)
        dt['Sample'] = pd.to_numeric(dt['Sample'], errors='coerce').astype(str)
        dt[decided_col] = pd.to_numeric(dt[decided_col], errors='coerce')

        cap = (dt[dt['Technologies'] == technology]
                 .dropna(subset=['Sample', decided_col])
                 .groupby('Sample', as_index=False)[decided_col].sum()
                 .rename(columns={decided_col: 'decided_capacity_GW'}))

        df = ct.merge(cap, on='Sample', how='left').fillna({'decided_capacity_GW': 0})

        if not plot:
            return df

        fig = px.scatter(
            df, x='commissioning_time', y='decided_capacity_GW',
            hover_data=['Sample'],
            title=f'{technology} — decided capacity vs commissioning time',
            labels={'commissioning_time': 'Commissioning time [years]',
                    'decided_capacity_GW': 'Total decided capacity [GW]'},
        )
        fig.update_traces(marker=dict(size=8, color='rgb(90,90,90)', opacity=0.7))
        pio.show(fig)

        outdir = self.outdir + "CapacityVsCommissioning/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + f"{technology}_capacity_vs_ct.html")
        fig.write_image(outdir + f"{technology}_capacity_vs_ct.pdf", width=900, height=600)
        plt.close()

        return df

    def capacity_by_commissioning_bin(self, technology, bin_size=0.5,
                                       ampl_uq_collector=None):
        """
        Capacité décidée moyenne par tranche de commissioning time (arrondi à bin_size).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        samples['Sample'] = samples['Sample'].astype(str)

        cp_col = f'cp_{technology}'
        if cp_col not in samples.columns:
            raise ValueError(f"'{cp_col}' introuvable dans Samples.")

        ct = samples[['Sample', cp_col]].copy()
        ct['ct_years'] = np.exp(pd.to_numeric(ct[cp_col], errors='coerce'))
        ct['ct_bin'] = (ct['ct_years'] / bin_size).round() * bin_size
        ct = ct.drop(columns=[cp_col])

        dt = ampl_uq_collector['Decision_tracking'].copy().reset_index()
        decided_col = next((c for c in ('F_decided_realized_up_to', 'F_decided_up_to')
                            if c in dt.columns), None)
        dt['Sample'] = pd.to_numeric(dt['Sample'], errors='coerce').astype(str)
        dt[decided_col] = pd.to_numeric(dt[decided_col], errors='coerce')

        cap = (dt[dt['Technologies'] == technology]
                 .dropna(subset=['Sample', decided_col])
                 .groupby('Sample', as_index=False)[decided_col].sum()
                 .rename(columns={decided_col: 'decided_capacity_GW'}))

        merged = ct.merge(cap, on='Sample', how='left').fillna({'decided_capacity_GW': 0})

        result = (merged.groupby('ct_bin')['decided_capacity_GW']
                        .agg(n_scenarios='count', mean_capacity='mean')
                        .round(2)
                        .reset_index()
                        .rename(columns={'ct_bin': f'ct_bin ({bin_size}y)'}))

        print(f"\n── {technology} — mean decided capacity [GW] by commissioning time bin ──")
        print(result.to_string(index=False))
        return result

    def graph_import_total_variation(self, ampl_uq_collector=None, plot=True,
                                      resources=None, year_start=None, year_end=None):
        """
        Box plot de la variation inter-scénarios de l'import total (somme sur toutes les années)
        pour AMMONIA, LFO, RE_METHANOL, RE_HYDROGEN, COAL [TWh].

        Parameters
        ----------
        resources : list | None
            Liste de noms de ressources. Par défaut : AMMONIA, LFO, METHANOL_RE, H2_RE, COAL.
        year_start / year_end : int | None
            Plage d'années à sommer (incluse).
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        _default = ['AMMONIA', 'LFO', 'METHANOL_RE', 'H2_RE', 'COAL', 'GAS']
        # Accept both naming conventions (user may type RE_METHANOL / RE_HYDROGEN)
        _aliases = {
            'RE_METHANOL': 'METHANOL_RE',
            'RE_HYDROGEN': 'H2_RE',
            'AMMONIA_RE':  'AMMONIA',
        }

        if resources is None:
            res_list = list(_default)
        elif isinstance(resources, str):
            res_list = [_aliases.get(resources, resources)]
        else:
            res_list = [_aliases.get(r, r) for r in resources]
        res_list = [str(r).strip() for r in res_list]

        def _canonicalize_resources(values):
            return (pd.Series(values, dtype='string')
                      .astype(str)
                      .str.strip()
                      .replace(_aliases))

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        if 'Resources' not in ampl_uq_collector:
            raise ValueError("'Resources' introuvable dans le collecteur UQ.")

        raw = ampl_uq_collector['Resources'].copy()

        # Filter by resource BEFORE reset_index to avoid column name ambiguity
        if isinstance(raw.index, pd.MultiIndex):
            res_level = next(
                (name for name in raw.index.names
                 if str(name).lower() not in ('years', 'year', 'sample')),
                None
            )
            yr_level = next(
                (name for name in raw.index.names
                 if str(name).lower() in ('years', 'year')),
                None
            )
            if res_level is not None:
                mask = _canonicalize_resources(
                    raw.index.get_level_values(res_level)
                ).isin(res_list).to_numpy()
                raw = raw[mask]
            if yr_level is not None:
                if year_start is not None:
                    raw = raw[raw.index.get_level_values(yr_level)
                                  .map(_year_key) >= int(year_start)]
                if year_end is not None:
                    raw = raw[raw.index.get_level_values(yr_level)
                                  .map(_year_key) <= int(year_end)]

        res = raw.reset_index()

        # Normalize column names
        col_map = {}
        for c in res.columns:
            cs = str(c).lower()
            if cs in ('years', 'year'):
                col_map[c] = 'Years'
            elif cs not in ('sample', 'res') and c not in col_map.values():
                col_map[c] = 'Resources'
        res = res.rename(columns=col_map)

        if 'Years' in res.columns:
            res['Years'] = res['Years'].astype(str).str.replace('YEAR_', '', regex=False)
        if 'Resources' not in res.columns:
            raise ValueError(f"Colonne ressource introuvable après reset_index. Colonnes: {list(res.columns)}")

        res['Resources'] = _canonicalize_resources(res['Resources'])

        # Apply filter also on column level (handles flat-index pickles)
        res = res[res['Resources'].astype(str).str.strip().isin(res_list)]

        if 'Years' in res.columns and year_start is not None:
            res = res[res['Years'].apply(_year_key) >= int(year_start)]
        if 'Years' in res.columns and year_end is not None:
            res = res[res['Years'].apply(_year_key) <= int(year_end)]

        res['Res'] = pd.to_numeric(res['Res'], errors='coerce').fillna(0) / 1000.0

        if res.empty:
            print("graph_import_total_variation: aucune donnée pour les ressources demandées.")
            print(f"  Ressources cherchées : {res_list}")
            return pd.DataFrame()

        # Sum over all years per (Sample, Resource)
        df_agg = (res.groupby(['Sample', 'Resources'], as_index=False)['Res']
                     .sum()
                     .loc[lambda d: d['Resources'].astype(str).isin(res_list)]
                     .copy())

        if df_agg.empty:
            print("graph_import_total_variation: aucune donnée après filtrage des ressources demandées.")
            print(f"  Ressources cherchées : {res_list}")
            return pd.DataFrame()

        order = [resource for resource in res_list if resource in df_agg['Resources'].astype(str).unique()]
        resource_totals = (df_agg.groupby('Resources')['Res']
                            .median()
                            .round(0)
                            .astype(int))

        if not plot:
            return df_agg

        fig = px.box(
            df_agg, x='Resources', y='Res',
            color='Resources',
            title='Total import variation across scenarios [TWh]',
            color_discrete_map=self.dict_color('Resources'),
            points='outliers', notched=False,
            category_orders={'Resources': order},
        )
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)
        pio.show(fig)

        outdir = self.outdir + "ImportVariation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/import_total_variation_raw.html")

        data_ymin = float(df_agg['Res'].min())
        data_ymax = float(df_agg['Res'].max())
        data_span = max(data_ymax - data_ymin, 1.0)
        y_pad = max(data_span * 0.08, 1.0)

        ymax = max(0.0, data_ymax)
        ymin = min(0.0, data_ymin)
        yvals = [round(ymin, 0), round(ymax, 0)]
        title_str = "<b>Total imports — variation across scenarios</b><br>[TWh]"
        if year_start or year_end:
            title_str = (f"<b>Total imports ({year_start or ''}–{year_end or ''}) "
                         f"— variation across scenarios</b><br>[TWh]")
        self.custom_fig(fig, title_str, yvals, type_graph='bar')
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
        fig.update_layout(showlegend=False)

        # Afficher uniquement les ticks min/max, avec l'axe à gauche,
        # les labels à gauche de l'axe et les ticks qui pointent vers le graphique
        y_tickvals = [float(v) for v in yvals]
        fig.update_yaxes(
            side='left',
            tickmode='array',
            tickvals=y_tickvals,
            ticktext=[f"{int(v):d}" for v in y_tickvals],
            ticks='inside',
            ticklabelposition='outside',
            automargin=True,
        )

        # Laisser une petite marge pour éviter que le tracé soit coupé en haut et en bas
        fig.update_yaxes(range=[data_ymin - y_pad, data_ymax + y_pad], autorange=False)

        # Axe Y visible et valeurs totales sous chaque ressource
        fig.update_yaxes(showline=True, linecolor='rgb(60,60,60)', linewidth=1.2)

        # Forcer les libellés des ressources sur l'axe X sans ligne d'axe
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(len(order))),
            ticktext=[label.lower() for label in order],
            tickangle=0,
            showline=False,
            linewidth=0,
            linecolor='rgba(0,0,0,0)',
            ticklabelposition='outside',
            showticklabels=True,
            tickfont=dict(size=14, family="Arial"),
        )

        y_min = float(df_agg['Res'].min())
        y_max = float(df_agg['Res'].max())
        y_span = max(y_max - y_min, 1.0)
        y_label_pos = y_min - (0.12 * y_span)

        for position, resource in enumerate(order):
            label = f"{resource_totals.loc[resource]:.0f}"
            fig.add_annotation(
                x=position,
                y=y_label_pos,
                xref='x',
                yref='y',
                text=label,
                showarrow=False,
                xanchor='center',
                yanchor='top',
                font=dict(size=18, color='rgb(70,70,70)'),
            )

        # Forcer les labels de l'axe X en horizontal avant export
        fig.update_xaxes(tickangle=0)

        # Superposer la valeur NCT (No commissioning) comme petite croix
        try:
            nct_path = '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/No commissioning/_Results.pkl'
            with open(nct_path, 'rb') as f:
                nct_results = pkl.load(f)

            nct_df = nct_results['Resources'].copy().reset_index()
            nct_df['Resources'] = _canonicalize_resources(nct_df['Resources'])
            nct_df['Res'] = pd.to_numeric(nct_df['Res'], errors='coerce').fillna(0) / 1000.0
            nct_totals = nct_df.groupby('Resources')['Res'].sum()

            nct_x = []
            nct_y = []
            for resource in order:
                value = nct_totals.get(resource)
                if pd.notna(value):
                    nct_x.append(resource)
                    nct_y.append(float(value))

            if nct_x and nct_y:
                fig.add_trace(go.Scatter(
                    x=nct_x,
                    y=nct_y,
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='rgb(180,60,60)', line=dict(width=2, color='rgb(180,60,60)')),
                    name='NCT',
                    showlegend=False,
                    hovertemplate='%{x}: %{y:.1f} TWh<extra>NCT</extra>',
                ))
        except Exception:
            pass

        # Format graphique plus agréable
        n = len(order)
        pdf_width = max(900, n * 180)
        pdf_height = 600
        fig.update_xaxes(range=[-0.5, n - 0.5], autorange=False)
        fig.update_layout(
            width=pdf_width,
            height=pdf_height,
            title_x=0.5,
            margin=dict(l=60, r=40, t=80, b=100),
            font=dict(size=22, family="Arial"),
            xaxis_title='',
            yaxis_title='[TWh]',
        )
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=20, family="Arial"))
        fig.update_yaxes(tickfont=dict(size=20, family="Arial"))
        fig.write_image(outdir + "import_total_variation.pdf", width=pdf_width, height=pdf_height)
        plt.close()

        return df_agg

    def graph_import_gas_coal_variation(self, ampl_uq_collector=None, plot=True,
                                         year_start=None, year_end=None):
        """
        Box plot de la variation inter-scénarios de l'import total de GAS et COAL [TWh],
        avec des croix pour les scénarios déterministe (DET) et sans commissioning (NCT).
        Min/max affichés sur l'axe Y.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        _resources = ['GAS', 'COAL']

        def _year_key(v):
            m = re.findall(r'\d+', str(v))
            return int(m[0]) if m else 0

        # ── Import UQ : même pattern que graph_resource_total_variation ─────────
        if 'Resources' not in ampl_uq_collector:
            raise ValueError("'Resources' introuvable dans le collecteur UQ.")

        res = ampl_uq_collector['Resources'].copy().reset_index()
        res['Years'] = res['Years'].astype(str).str.replace('YEAR_', '', regex=False)
        res['Res'] = pd.to_numeric(res['Res'], errors='coerce').fillna(0) / 1000.0
        if year_start is not None:
            res = res[res['Years'].apply(_year_key) >= int(year_start)]
        if year_end is not None:
            res = res[res['Years'].apply(_year_key) <= int(year_end)]

        # Groupby d'abord (comme graph_resource_total_variation), filtre après
        df_agg = res.groupby(['Sample', 'Resources'], as_index=False)['Res'].sum()
        df_agg = df_agg[df_agg['Resources'].astype(str).str.strip().isin(_resources)]

        if df_agg.empty:
            available = res['Resources'].astype(str).unique().tolist()
            print(f"graph_import_gas_coal_variation: GAS/COAL introuvables.")
            print(f"  Ressources disponibles : {available[:20]}")
            return pd.DataFrame()

        # ── Valeurs DET / NCT ─────────────────────────────────────────────────
        det_paths = {
            'DET': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/Deterministic (mean value)/_Results.pkl',
            'NCT': '/opt/anaconda3/envs/env_arm/lib/python3.11/site-packages/rheia/RESULTS/UNC_ANAL_ES_PATHWAY/Deterministic/No commissioning/_Results.pkl',
        }
        det_colors  = {'DET': 'rgb(60,60,60)',  'NCT': 'rgb(180,60,60)'}
        det_symbols = {'DET': 'x',              'NCT': 'x-open'}

        def _get_det_totals(pkl_path):
            totals = {}
            try:
                with open(pkl_path, 'rb') as f:
                    det_res = pkl.load(f)
                if 'Resources' not in det_res:
                    return totals
                det_r = det_res['Resources'].copy().reset_index()
                det_r['Years'] = det_r['Years'].astype(str).str.replace('YEAR_', '', regex=False)
                det_r['Res'] = pd.to_numeric(det_r['Res'], errors='coerce').fillna(0) / 1000.0
                if year_start is not None:
                    det_r = det_r[det_r['Years'].apply(_year_key) >= int(year_start)]
                if year_end is not None:
                    det_r = det_r[det_r['Years'].apply(_year_key) <= int(year_end)]
                det_r = det_r[det_r['Resources'].astype(str).isin(_resources)]
                for r, grp in det_r.groupby('Resources'):
                    totals[str(r)] = float(grp['Res'].sum())
            except Exception:
                pass
            return totals

        det_totals = {lbl: _get_det_totals(p) for lbl, p in det_paths.items()}

        order = (df_agg.groupby('Resources')['Res']
                       .median()
                       .sort_values(ascending=False)
                       .index.tolist())

        if not plot:
            return df_agg

        # ── Figure ────────────────────────────────────────────────────────────
        fig = px.box(
            df_agg, x='Resources', y='Res',
            color='Resources',
            color_discrete_map=self.dict_color('Resources'),
            points='outliers', notched=False,
            category_orders={'Resources': order},
        )

        for label, totals in det_totals.items():
            if not totals:
                continue
            xs = [r for r in order if r in totals]
            ys = [totals[r] for r in xs]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='markers',
                name=label,
                marker=dict(size=14, color=det_colors[label],
                            symbol=det_symbols[label], line=dict(width=2)),
                showlegend=True,
            ))

        pio.show(fig)

        # ── Export PDF ────────────────────────────────────────────────────────
        outdir = self.outdir + "ImportVariation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        fig.write_html(outdir + "_Raw/import_gas_coal_variation_raw.html")

        all_vals = list(df_agg['Res'])
        for totals in det_totals.values():
            all_vals += list(totals.values())
        ymin = min(0.0, float(min(all_vals)))
        ymax = max(0.0, float(max(all_vals)))
        yvals = [round(ymin, 0), round(ymax, 0)]

        title_str = "<b>Total GAS & COAL imports — variation across scenarios</b><br>[TWh]"
        self.custom_fig(fig, title_str, yvals, xvals=order, type_graph='bar')
        # Forcer les ticks min/max après custom_fig (qui peut écraser tickvals)
        fig.update_yaxes(
            tickmode='array',
            tickvals=yvals,
            ticktext=[str(int(v)) for v in yvals],
        )
        fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=0)
        fig.update_layout(
            showlegend=True,
            legend=dict(x=0.99, y=0.99, xanchor='right', yanchor='top',
                        bgcolor='rgba(255,255,255,0.85)',
                        bordercolor='rgba(90,90,90,0.3)', borderwidth=1),
        )
        fig.write_image(outdir + "import_gas_coal_variation.pdf", width=900, height=550)
        plt.close()

        return df_agg

    def graph_import_commissioning_correlation(self, ampl_uq_collector=None, plot=True,
                                                resources=None, year_start=None, year_end=None,
                                                method='spearman'):
        """
        Heatmap de corrélation entre les paramètres de commissioning time (cp_*)
        et l'import total (somme sur toutes les années) de AMMONIA, LFO, RE_METHANOL,
        RE_HYDROGEN, COAL, GAS [TWh].

        Parameters
        ----------
        resources : list | None
            Ressources à analyser. Par défaut : AMMONIA, LFO, METHANOL_RE, H2_RE, COAL, GAS.
        year_start / year_end : int | None
            Filtre sur les années.
        method : str
            'spearman' (défaut) ou 'pearson'.
        """
        if ampl_uq_collector is None:
            ampl_uq_collector = self.ampl_uq_collector

        # ── Résoudre les alias (même logique que graph_import_total_variation) ─
        _default = ['AMMONIA', 'LFO', 'METHANOL_RE', 'H2_RE', 'COAL', 'GAS']
        _aliases = {'RE_METHANOL': 'METHANOL_RE', 'RE_HYDROGEN': 'H2_RE', 'AMMONIA_RE': 'AMMONIA'}
        canonical = [_aliases.get(r, r) for r in resources] if resources is not None else _default

        # ── Import totaux par scénario ─────────────────────────────────────────
        df_agg = self.graph_import_total_variation(
            ampl_uq_collector=ampl_uq_collector, plot=False,
            resources=resources, year_start=year_start, year_end=year_end,
        )
        if df_agg.empty:
            print("graph_import_commissioning_correlation: aucune donnée d'import.")
            return pd.DataFrame()

        # Garder uniquement les ressources demandées (garde-fou contre fuites)
        df_agg = df_agg[df_agg['Resources'].astype(str).isin(canonical)]

        # ── Commissioning time (colonnes cp_*) ────────────────────────────────
        samples = ampl_uq_collector['Samples'].copy().reset_index(drop=True)
        if 'Sample' not in samples.columns:
            samples['Sample'] = np.arange(1, len(samples) + 1)
        samples['Sample'] = samples['Sample'].astype(str)

        _ct_exclude = {'cp_CCGT', 'cp_NUCLEAR_SMR'}
        ct_cols = [c for c in samples.columns
                   if isinstance(c, str) and c.startswith('cp_') and c not in _ct_exclude]
        if not ct_cols:
            raise ValueError("Aucune colonne cp_* de commissioning time trouvée dans Samples.")

        df_ct = samples[['Sample'] + ct_cols].copy()
        df_ct['Sample'] = df_ct['Sample'].astype(str)

        # ── Pivot imports: lignes = Sample, colonnes = Resources ──────────────
        df_agg['Sample'] = df_agg['Sample'].astype(str)
        df_pivot = df_agg.pivot_table(index='Sample', columns='Resources',
                                      values='Res', aggfunc='sum')
        # Conserver l'ordre de canonical pour les ressources présentes dans le pivot
        ordered_res = [r for r in canonical if r in df_pivot.columns]
        df_pivot = df_pivot[ordered_res]

        df_merged = df_ct.set_index('Sample').join(df_pivot, how='inner')
        if df_merged.empty:
            raise ValueError("Aucun sample commun entre Samples et les données d'import.")

        res_names = list(df_pivot.columns)
        corr_matrix = pd.DataFrame(index=res_names, columns=ct_cols, dtype=float)

        for res_name in res_names:
            for ct in ct_cols:
                s = df_merged[[ct, res_name]].dropna()
                if len(s) < 3:
                    corr_matrix.loc[res_name, ct] = float('nan')
                    continue
                if method == 'spearman':
                    r = s[ct].rank().corr(s[res_name].rank())
                else:
                    r = s[ct].corr(s[res_name])
                corr_matrix.loc[res_name, ct] = round(float(r), 3)

        # Trier par corrélation max absolue (les plus sensibles en haut)
        corr_matrix['_max'] = corr_matrix[ct_cols].abs().max(axis=1)
        corr_matrix.sort_values('_max', ascending=False, inplace=True)
        corr_matrix.drop(columns='_max', inplace=True)

        # Supprimer les lignes sans aucune corrélation >= 0.3
        mask_keep = corr_matrix[ct_cols].abs().max(axis=1) >= 0.3
        corr_matrix = corr_matrix.loc[mask_keep]

        if corr_matrix.empty:
            print("graph_import_commissioning_correlation: aucune corrélation ≥ 0.3.")
            return corr_matrix

        ct_labels  = [self.uncert_param_meaning.get(c, c) for c in ct_cols]
        res_labels = list(corr_matrix.index)

        if not plot:
            return corr_matrix

        outdir = self.outdir + "ImportVariation/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir + '_Raw/').mkdir(parents=True, exist_ok=True)
        corr_matrix.to_csv(outdir + "corr_import_ct.csv")

        title = f"Correlation — commissioning time vs total imports ({method})"
        self._export_corr_heatmap(
            row_matrix=corr_matrix,
            x_labels=ct_labels,
            y_labels=res_labels,
            title=title,
            zmin=-1, zmax=1,
            out_dir=outdir,
            filename_stem="corr_import_ct",
            decimals=2,
            interpolation='nearest',
            text_threshold=0.3,
            xlabel_rotation=0,
        )
        return corr_matrix

    def _get_heat_technologies(self, layers=None):
        """Retourne les technologies de chaleur depuis dict_color."""
        high_t = list(self.dict_color('Heat_high_T').keys())
        dhn    = [t for t in self.dict_color('Heat_low_T').keys() if t.startswith('DHN_')]
        decen  = [t for t in self.dict_color('Heat_low_T').keys() if t.startswith('DEC_')]
        layer_map = {
            'HEAT_HIGH_T': high_t,
            'HEAT_LOW_T_DHN': dhn,
            'HEAT_LOW_T_DECEN': decen,
        }
        if layers is None:
            return high_t + dhn + decen
        techs = []
        for lyr in layers:
            techs.extend(layer_map.get(lyr, []))
        return list(dict.fromkeys(techs))


    def _get_electricity_technologies(self):
        if self.ampl_obj is not None and hasattr(self.ampl_obj, 'sets'):
            tech_map = self.ampl_obj.sets.get('TECHNOLOGIES_OF_END_USES_TYPE', {})
            elec_tech = tech_map.get('ELECTRICITY', [])
            if len(elec_tech) > 0:
                return list(elec_tech)

        fallback = list(self.dict_color('Electricity').keys())
        fallback = [t for t in fallback if t != 'ELECTRICITY']
        return fallback


    @staticmethod
    def _fmt_tech_label(raw):
        """'IND_COGEN_GAS' → 'IND Cogen gas'  (first token uppercase, rest sentence-case)."""
        parts = str(raw).split('_')
        if len(parts) == 1:
            return raw
        return parts[0].upper() + ' ' + ' '.join(parts[1:]).lower().capitalize()

    @staticmethod
    def _phase_sort_key(phase):
        text = str(phase)
        nums = [int(x) for x in re.findall(r'\d+', text)]
        if len(nums) == 0:
            return (10**9, text)
        return (nums[0], text)


    @staticmethod
    def _phase_center_numeric(phase):
        text = str(phase)
        nums = [int(x) for x in re.findall(r'\d+', text)]
        if len(nums) >= 2:
            return 0.5 * (nums[0] + nums[1])
        if len(nums) == 1:
            return float(nums[0])
        return 0.0
    
    
    def _fill_df_to_plot_w_zeros(self,df_to_plot):
        index_names = df_to_plot.index.names
        l_ind = [None] * len(index_names)
        for i,j in enumerate(index_names):
            ind = df_to_plot.index.get_level_values(j).unique()
            if j == 'Years' and not('YEAR_2020' in ind):
                ind = pd.Index(list(ind) + ['YEAR_2020'])
            l_ind[i] = ind
        
        mi_temp = pd.MultiIndex.from_product(l_ind,names=index_names)
        df_temp = pd.DataFrame(0,index=mi_temp,columns=df_to_plot.columns)
        df_temp.update(df_to_plot)
        df_to_plot = df_temp.copy()
        return df_to_plot
        
        
        
    def _polyfit_func(self,x_in, y_in, threshold=0.99999999):
        """
        The function fits a polynomial to the points of x_in and y_in. The
        polynomial starts with order 1. To evaluate its performance, the R-squared
        performance indicator is quantified. If the value for R-squared does
        not reach the defined threshold, the polynomial order is increased and
        the polynomial is fitted again on the points, until the threshold is
        satisfied. Once satisfied, the function returns the polynomial.
    
        Parameters
        ----------
        x_in : ndarray
            The x-coordinates for the sample points.
        y_in : ndarray
            The y-coordinates for the sample points.
        threshold : float, optional
            The threshold for the R-squared parameter. The default is 0.99999999.
    
        Returns
        -------
        poly_func : numpy.poly1d
            A one-dimensional polynomial.
    
        """
        order = 0
        r_squared = 0.
        while r_squared < threshold and order < 15:
            order += 1
    
            # the polynomial
            poly_coeff = np.polyfit(x_in, y_in, order)
            poly_func = np.poly1d(poly_coeff)
    
            # r-squared
            yhat = poly_func(x_in)
            ybar = np.sum(y_in) / len(y_in)
            ssreg = np.sum((yhat - ybar)**2.)
            sstot = np.sum((y_in - ybar)**2.)
            r_squared = ssreg / sstot
    
        return poly_func

    def _remove_low_values(self,df_temp, threshold = None):
        if threshold == None:
            threshold = self.threshold_filter
        max_temp = np.nanmax(df_temp)
        min_temp = np.nanmin(df_temp)
        df_temp = df_temp.loc[(abs(df_temp) > 1e-8) & ((df_temp >= threshold*max_temp) |
                              (df_temp <= threshold*min_temp))]
        return df_temp
    
    def _group_tech_per_eud(self):
        tech_of_end_uses_category = self.ampl_obj.sets['TECHNOLOGIES_OF_END_USES_CATEGORY'].copy()
        tech_of_end_uses_category.pop("NON_ENERGY", None)
        end_uses_types = self.ampl_obj.sets.get('END_USES_TYPES_OF_CATEGORY', {})

        if 'NON_ENERGY' in end_uses_types:
            for ned in end_uses_types['NON_ENERGY']:
                tech_of_end_uses_category[ned] = self.ampl_obj.sets['TECHNOLOGIES_OF_END_USES_TYPE'][ned]

        df = tech_of_end_uses_category.copy()
        df['INFRASTRUCTURE'] = self.ampl_obj.sets['INFRASTRUCTURE']
        df['STORAGE'] = self.ampl_obj.sets['STORAGE_TECH']
        
        return df
    
    def _group_sets(self):
        categories = self._group_tech_per_eud()
        categories['STORAGE'] = self.ampl_obj.sets['STORAGE_TECH'].copy()
        categories['RE_FUELS'] = self.ampl_obj.sets['RE_RESOURCES'].copy()
        categories['INFRASTRUCTURE'] = self.ampl_obj.sets['INFRASTRUCTURE'].copy()
        re_fuels = self.ampl_obj.sets['RE_RESOURCES'].copy()
        resources = self.ampl_obj.sets['RESOURCES'].copy()
        nre_fuels = [res for res in resources if res not in re_fuels]
        categories['NRE_FUELS'] = nre_fuels
        
        categories_2 = dict()
        
        for k in categories:
            for j in categories[k]:
                if k == 'HEAT_LOW_T':
                    if j in self.ampl_obj.sets['TECHNOLOGIES_OF_END_USES_TYPE']["HEAT_LOW_T_DHN"]:
                        categories_2[j] = 'HEAT_LOW_T_DHN'
                    else:
                        categories_2[j] = 'HEAT_LOW_T_DECEN'
                elif k == 'MOBILITY_PASSENGER':
                    if j in self.ampl_obj.sets['TECHNOLOGIES_OF_END_USES_TYPE']["MOB_PUBLIC"]:
                        categories_2[j] = 'MOB_PUBLIC'
                    else:
                        categories_2[j] = 'MOB_PRIVATE'
                else :
                    categories_2[j] = k
        
        return categories_2
    
    
    def _dict_color_full(self):
        year_balance = self.ampl_uq_collector['Year_balance']
        elements = year_balance.index.get_level_values(1).unique()
        color_dict_full = dict.fromkeys(elements)
        categories = ['Sectors','Electricity','Heat_low_T','Heat_high_T','Mobility','Freight','Ammonia',
                   'Methanol','HVC','Conversion','Storage','Storage','Storage_daily','Resources',
                   'Infrastructure','Years_1','Years_2','Phases']
        for c in categories:
            color_dict_full.update(self.dict_color(c))
        
        color_dict_full['END_USES'] = 'lightsteelblue'
        
        return color_dict_full
    
    @staticmethod
    def dict_color(category):
        color_dict = {}
        
        if category == 'Electricity':
            color_dict = {"NUCLEAR":"deeppink", "NUCLEAR_SMR": "pink", "CCGT":"darkorange", "CCGT_AMMONIA":"slateblue", "COAL_US" : "black", "COAL_IGCC" : "dimgray", "PV_RESIDENTIAL" : "yellow", "PV_FIELD" : "gold", "WIND_ONSHORE" : "lawngreen", "WIND_OFFSHORE" : "green", "HYDRO_RIVER" : "blue", "GEOTHERMAL" : "firebrick", "ELECTRICITY" : "dodgerblue"}
        elif category == 'Heat_low_T':
            color_dict = {"DHN_HP_ELEC" : "blue", "DHN_COGEN_GAS" : "orange", "DHN_COGEN_WOOD" : "sandybrown", "DHN_COGEN_WASTE" : "olive", "DHN_COGEN_WET_BIOMASS" : "seagreen", "DHN_COGEN_BIO_HYDROLYSIS" : "springgreen", "DHN_BOILER_GAS" : "darkorange", "DHN_BOILER_WOOD" : "sienna", "DHN_BOILER_OIL" : "blueviolet", "DHN_DEEP_GEO" : "firebrick", "DHN_SOLAR" : "gold", "DEC_HP_ELEC" : "cornflowerblue", "DEC_THHP_GAS" : "lightsalmon", "DEC_COGEN_GAS" : "goldenrod", "DEC_COGEN_OIL" : "mediumpurple", "DEC_ADVCOGEN_GAS" : "burlywood", "DEC_ADVCOGEN_H2" : "violet", "DEC_BOILER_GAS" : "moccasin", "DEC_BOILER_WOOD" : "peru", "DEC_BOILER_OIL" : "darkorchid", "DEC_SOLAR" : "yellow", "DEC_DIRECT_ELEC" : "deepskyblue"}
        elif category == 'Heat_high_T':
            color_dict = {"IND_COGEN_GAS":"orange", "IND_COGEN_WOOD":"peru", "IND_COGEN_WASTE" : "olive", "IND_BOILER_GAS" : "moccasin", "IND_BOILER_WOOD" : "goldenrod", "IND_BOILER_OIL" : "blueviolet", "IND_BOILER_COAL" : "black", "IND_BOILER_WASTE" : "olivedrab", "IND_DIRECT_ELEC" : "royalblue"}
        elif category == 'Mobility':
            color_dict = {"TRAMWAY_TROLLEY" : "dodgerblue", "BUS_COACH_DIESEL" : "dimgrey", "BUS_COACH_HYDIESEL" : "gray", "BUS_COACH_CNG_STOICH" : "orange", "BUS_COACH_FC_HYBRIDH2" : "violet", "TRAIN_PUB" : "blue", "CAR_GASOLINE" : "black", "CAR_DIESEL" : "lightgray", "CAR_NG" : "moccasin", "CAR_METHANOL":"orchid", "CAR_HEV" : "salmon", "CAR_PHEV" : "lightsalmon", "CAR_BEV" : "deepskyblue", "CAR_FUEL_CELL" : "magenta"}
        elif category == 'Freight':
            color_dict = {"TRAIN_FREIGHT" : "royalblue", "BOAT_FREIGHT_DIESEL" : "dimgrey", "BOAT_FREIGHT_NG" : "darkorange", "BOAT_FREIGHT_METHANOL" : "fuchsia", "TRUCK_DIESEL" : "darkgrey", "TRUCK_FUEL_CELL" : "violet", "TRUCK_ELEC" : "dodgerblue", "TRUCK_NG" : "moccasin", "TRUCK_METHANOL" : "orchid"}
        elif category == 'Ammonia':
            color_dict = {"HABER_BOSCH":"tomato", "AMMONIA" : "slateblue", "AMMONIA_RE" : "blue"}
        elif category == 'Methanol':
            color_dict = {"SYN_METHANOLATION":"violet","METHANE_TO_METHANOL":"orange","BIOMASS_TO_METHANOL":"peru", "METHANOL" : "orchid", "METHANOL_RE" : "mediumvioletred"}
        elif category == "HVC":
            color_dict = {"OIL_TO_HVC":"blueviolet", "GAS_TO_HVC":"orange", "BIOMASS_TO_HVC":"peru", "METHANOL_TO_HVC":"orchid"}
        elif category == 'Conversion':
            color_dict = {"H2_ELECTROLYSIS" : "violet", "H2_NG" : "magenta", "H2_BIOMASS" : "orchid", "GASIFICATION_SNG" : "orange", "PYROLYSIS" : "blueviolet", "ATM_CCS" : "black", "INDUSTRY_CCS" : "grey", "SYN_METHANOLATION" : "mediumpurple", "SYN_METHANATION" : "moccasin", "BIOMETHANATION" : "darkorange", "BIO_HYDROLYSIS" : "gold", "METHANE_TO_METHANOL" : "darkmagenta",'SMR':'orange', 'AMMONIA_TO_H2':'fuchsia'}
        elif category == 'Storage':
            color_dict = {"TS_DHN_SEASONAL" : "indianred", "BATT_LI" : "royalblue", "BEV_BATT" : "deepskyblue", "PHEV_BATT" : "lightskyblue", "PHS" : "dodgerblue", "TS_DEC_HP_ELEC" : "blue", "TS_DHN_DAILY" : "lightcoral", "TS_HIGH_TEMP" : "red", "SEASONAL_NG" : "orange", "SEASONAL_H2" : "violet", "SLF_STO" : "blueviolet", "TS_DEC_DIRECT_ELEC":"darkgoldenrod", "TS_DEC_THHP_GAS": "orange", "TS_DEC_COGEN_GAS":"coral", "TS_DEC_COGEN_OIL":"darkviolet", "TS_DEC_ADVCOGEN_GAS":"sandybrown", "TS_DEC_ADVCOGEN_H2": "plum", "TS_DEC_BOILER_GAS": "tan", "TS_DEC_BOILER_WOOD":"peru", "TS_DEC_BOILER_OIL": "darkviolet", "GAS_STORAGE": "orange", "H2_STORAGE": "violet", "CO2_STORAGE": "lightgray", "GASOLINE_STORAGE": "gray", "DIESEL_STORAGE": "silver", "AMMONIA_STORAGE": "slateblue", "LFO_STORAGE": "darkviolet"}
        elif category == 'Storage_daily':
            color_dict = {"BATT_LI" : "royalblue", "BEV_BATT" : "deepskyblue", "PHEV_BATT" : "lightskyblue", "TS_DEC_HP_ELEC" : "blue", "TS_DHN_DAILY" : "lightcoral", "TS_HIGH_TEMP" : "red", "TS_DEC_DIRECT_ELEC":"darkgoldenrod", "TS_DEC_THHP_GAS": "orange", "TS_DEC_COGEN_GAS":"coral", "TS_DEC_COGEN_OIL":"darkviolet", "TS_DEC_ADVCOGEN_GAS":"sandybrown", "TS_DEC_ADVCOGEN_H2": "plum", "TS_DEC_BOILER_GAS": "tan", "TS_DEC_BOILER_WOOD":"peru", "TS_DEC_BOILER_OIL": "darkviolet"}
        elif category == 'Resources':
            color_dict = {"ELECTRICITY" : "deepskyblue", "GASOLINE" : "gray", "DIESEL" : "silver", "BIOETHANOL" : "mediumorchid", "BIODIESEL" : "mediumpurple", "LFO" : "darkviolet", "GAS" : "orange", "GAS_RE" : "gold", "WOOD" : "saddlebrown", "WET_BIOMASS" : "seagreen", "COAL" : "black", "URANIUM" : "deeppink", "WASTE" : "olive", "H2" : "violet", "H2_RE" : "plum", "AMMONIA" : "slateblue", "AMMONIA_RE" : "blue", "METHANOL" : "orchid", "METHANOL_RE" : "mediumvioletred", "CO2_EMISSIONS" : "gainsboro", "RES_WIND" : "limegreen", "RES_SOLAR" : "yellow", "RES_HYDRO" : "blue", "RES_GEO" : "firebrick", "ELEC_EXPORT" : "chartreuse","CO2_ATM": "dimgray", "CO2_INDUSTRY": "darkgrey", "CO2_CAPTURED": "lightslategrey", "RE_FUELS": 'green','NRE_FUELS':'black', 'LOCAL_RE': 'limegreen', 'IMPORTED_ELECTRICITY': 'deepskyblue'}
        elif category == 'Sectors':
            color_dict = {"ELECTRICITY" : "deepskyblue", "HEAT_HIGH_T":"red","HEAT_LOW_T_DECEN":"lightpink", "HEAT_LOW_T_DHN":"indianred", "MOB_PUBLIC":"gold", "MOB_PRIVATE":"goldenrod","MOBILITY_FREIGHT":"darkgoldenrod", "NON_ENERGY": "darkviolet", "INFRASTRUCTURE":"grey","HVC":"cyan",'STORAGE':'chartreuse', 'OTHERS':'gainsboro'}
        elif category == 'Infrastructure':
            color_dict = {'EFFICIENCY': 'lime','DHN': 'orange','GRID': 'gold'}
        elif category == 'Years_1':
            color_dict = {'YEAR_2020': 'blue','YEAR_2025': 'orange','YEAR_2030': 'green', 'YEAR_2035': 'red', 'YEAR_2040': 'purple', 'YEAR_2045': 'brown', 'YEAR_2050':'pink'}
        elif category == 'Years_2':
            color_dict = {'2020': 'blue','2025': 'orange','2030': 'green', '2035': 'red', '2040': 'purple', '2045': 'brown', '2050':'pink'}
        elif category == 'Phases':
                color_dict = {'2015_2020': 'blue','2020_2025': 'orange','2025_2030': 'green', '2030_2035': 'red', '2035_2040': 'purple', '2040_2045': 'brown', '2045_2050':'pink'}
            
        return color_dict
    
    
    def dict_meaning(self):
        meaning_dict = {'H2_RE':'E-hydrogen',
                        'AMMONIA_RE': 'E-ammonia',
                        'METHANOL_RE': 'E-methanol',
                        'GAS_RE': 'E-methane',
                        'H2':'Fossil hydrogen',
                        'AMMONIA': 'Fossil ammonia',
                        'METHANOL': 'Fossil methanol',
                        'GAS': 'Fossil methane',
                        'H2_ELECTROLYSIS': 'Electrolyser',
                        'CCGT_AMMONIA': 'Ammonia CCGT',
                        'SYN_METHANOLATION': 'Methanolation',
                        'METHANE_TO_METHANOL': 'Methane-to-methanol',
                        'NUCLEAR_SMR': 'Nuclear SMR',
                        'BIOMETHANATION': 'Biomethanation',
                        'BIO_HYDROLYSIS': 'Biohydrolysis',
                        'Total gwp' : 'Total gwp',
                        'PV_RESIDENTIAL': 'PV residential',
                        'PV_FIELD': 'PV field',
                        'WIND_ONSHORE': 'Onshore wind',
                        'WIND_OFFSHORE': 'Offshore wind',
                        'GEOTHERMAL': 'Geothermal',
                        'HYDRO_RIVER': 'Hydro river',
                        'NUCLEAR': 'Nuclear',
                        'SMR': 'Steam-methane-reforming',
                        'AMMONIA_TO_H2':'Ammonia-to-H2',
                        'CAR_FUEL_CELL':'Fuel cell car'
                        }
        
        return meaning_dict
        
    
    @staticmethod
    def custom_fig(fig,title,yvals,xvals=list(range(2020, 2051)), ftsize=18,annot_text=None,
                   type_graph = None, neg_value = False, flip = False,
                   x_unit=None, y_unit=None):
    
        def round_repdigit(n, ndigits=0):     
            if n != 0:
                i = int(np.ceil(np.log10(abs(n))))
                x = np.round(n, ndigits-i)
                if i-ndigits >= 0:
                    x = int(x)
                return x     
            else:
                return 0
            
        gray = 'rgb(90,90,90)' 
        color = gray
        
        fig.update_layout(
            xaxis_color=color, yaxis_color=color,
            xaxis_mirror=False, yaxis_mirror=False,
            yaxis_showgrid=False, xaxis_showgrid=False,
            yaxis_linecolor='white', xaxis_linecolor='white',
            xaxis_tickfont_size=ftsize, yaxis_tickfont_size=ftsize,
            showlegend=False,
        )
        
        parsed_title_unit = None

        if title is not None:
            title_parts = str(title).split('<br>', 1)
            title_main = re.sub(r'</?b>', '', title_parts[0]).strip()
            if len(title_parts) > 1:
                parsed_title_unit = title_parts[1].strip()

            fig.update_layout(
                title=dict(
                    text=title_main,
                    x=0.5,
                    xanchor='center',
                    font=dict(family="Raleway", size=ftsize+10)
                )
            )

        def _clean_unit_text(unit_text):
            if unit_text is None:
                return None
            text = str(unit_text).strip()
            return text if text else None

        # Backward compatibility: old titles encoded y-unit as "<br>[unit]".
        x_unit_text = _clean_unit_text(x_unit)
        y_unit_text = _clean_unit_text(y_unit)
        if y_unit_text is None and parsed_title_unit is not None:
            y_unit_text = _clean_unit_text(re.sub(r'</?b>', '', parsed_title_unit))
        
        if type_graph == None:
            fig.update_xaxes(dict(ticks = "inside", ticklen=10))
            fig.update_xaxes(tickangle= 0,tickmode = 'array',tickwidth=2,tickcolor=gray,
                                tickfont=dict(
                                      family="Rawline",
                                      size=ftsize
                                  ))
        fig.update_yaxes(dict(ticks = "inside", ticklen=10))
        if type_graph in ['strip','bar']:
            fig.update_xaxes(dict(ticks = "inside", ticklen=10, tickangle=0))
            
        
        if not(flip):
            fig.update_yaxes(tickangle= 0,tickmode = 'array',tickwidth=2,tickcolor=gray,
                                tickfont=dict(
                                      family="Rawline",
                                      size=ftsize
                                  ))
        
        fig.update_layout(
            yaxis = dict(
                tickmode = 'array',
                tickvals = fig.layout.yaxis.tickvals,
                ticktext = list(map(str,yvals))
                ))
        
                
        factor=0.05
        nrepdigit = 0
        
        fig.update_yaxes(tickvals=yvals)
        
        xstring = isinstance(fig.data[0].x[0],str)
        
        if xstring: ## ONLY VALID IF THE FIRST TRACE HAS ALL VALUES
            xvals = xvals
            xmin = 0
            xmax = len(xvals) - 1
        else:
            if not(flip):
                xvals = pd.Series(sum([list(i.x) for i in fig.data],[]))
                xmin = xvals.min()
                xmax = xvals.max()
            else:
                xmin = min(xvals)
                xmax = max(xvals)
            xampl = xmax-xmin
        
        
        if xstring:
            if type_graph in ['bar','strip']:
                fig.layout.xaxis.range = [xmin-factor*15, xmax+factor*15]
            elif type_graph == 'scatter':
                fig.layout.xaxis.range = [xmin-factor*6, xmax+factor*6]
            else:
                fig.layout.xaxis.range = [xmin-factor*3, xmax+factor*3]
            if fig.layout.xaxis.tickvals is None:
                fig.layout.xaxis.tickvals = xvals
        else:
            if fig.layout.xaxis.range is None:
                fig.layout.xaxis.range = [xmin-xampl*factor, xmax+xampl*factor]
            if fig.layout.xaxis.tickvals is None:
                fig.layout.xaxis.tickvals = [round_repdigit(x, nrepdigit) for x in [xmin, xmax]]
        
        if flip:
            fig.update_xaxes(tickvals=xvals)
            
        
        fig.layout.xaxis.tickvals = sorted(fig.layout.xaxis.tickvals)
        fig.layout.xaxis.range = sorted(fig.layout.xaxis.range)
        
        
        yvals = yvals
        
        ystring = isinstance(fig.data[0].y[0],str)
        
        if ystring: ## ONLY VALID IF THE FIRST TRACE HAS ALL VALUES
            ymin = 0
            ymax = len(yvals) - 1
        else:
            ymin = min(yvals)
            ymax = max(yvals)
        yampl = ymax-ymin
        
        if fig.layout.yaxis.range is None:
            fig.layout.yaxis.range = [ymin-yampl*factor, ymax+yampl*factor]
        if fig.layout.yaxis.tickvals is None:
            fig.layout.yaxis.tickvals = [round_repdigit(y, nrepdigit) for y in [ymin, ymax]]
        
        if not(ystring):
            fig.layout.yaxis.tickvals = sorted(fig.layout.yaxis.tickvals)
        fig.layout.yaxis.range = sorted(fig.layout.yaxis.range)
        
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        
        if neg_value:
            if not(flip):
                fig.add_shape(x0=xmin,x1=xmax,
                          y0=0,y1=0,
                          type='line',layer="below",
                          line=dict(color=color,width=1),opacity=0.5)
            else:
                fig.add_shape(x0=0,x1=0,
                          y0=ymin,y1=ymax,
                          type='line',layer="below",
                          line=dict(color=color,width=1),opacity=0.5)
                
        
        fig.add_shape(x0=fig.layout.xaxis.range[0],x1=fig.layout.xaxis.range[0],
                  y0=fig.layout.yaxis.tickvals[0],y1=fig.layout.yaxis.tickvals[-1],
                  type='line',layer="above",
                  line=dict(color=color,width=2),opacity=1)
        
        # if type_graph in [None,'bar']:
        fig.add_shape(x0=xmin,x1=xmax,
                  y0=fig.layout.yaxis.range[0],y1=fig.layout.yaxis.range[0],
                  type='line',layer="above",
                  line=dict(color=color, width=2),opacity=1)
        
        if type_graph in ['scatter']:
            fig.update_layout({ax:{"visible":False, "matches":None} for ax in fig.to_dict()["layout"] if "xaxis" in ax})
        
        if type_graph in ['strip']:
            if not(flip):
                fig.add_shape(x0=0.5,x1=0.5,
                          y0=ymin,y1=ymax,
                          type='line',layer="above",
                          line=dict(color=color,width=2,dash="dot"),opacity=1)
                fig.add_shape(x0=xmax-0.5,x1=xmax-0.5,
                          y0=ymin,y1=ymax,
                          type='line',layer="above",
                          line=dict(color=color,width=2,dash="dot"),opacity=1)
            # else:
                # fig.add_shape(x0=xmin,x1=xmax,
                #           y0=0.5,y1=0.5,
                #           type='line',layer="above",
                #           line=dict(color=color,width=2,dash="dot"),opacity=1)
                # fig.add_shape(x0=xmin,x1=xmax,
                #           y0=ymax-0.5,y1=ymax-0.5,
                #           type='line',layer="above",
                #           line=dict(color=color,width=2,dash="dot"),opacity=1)
        
        if y_unit_text:
            fig.add_annotation(
                xref='paper',
                yref='paper',
                x=0,
                y=1.02,
                text=y_unit_text,
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=ftsize, color=gray)
            )

        if x_unit_text:
            fig.add_annotation(
                xref='paper',
                yref='paper',
                x=1,
                y=-0.10,
                text=x_unit_text,
                showarrow=False,
                xanchor='right',
                yanchor='top',
                font=dict(size=ftsize, color=gray)
            )

        margin_bottom = 45 if x_unit_text else 10
        fig.update_layout(margin_b = margin_bottom, margin_r = 30, margin_l = 30)#,margin_pad = 20)