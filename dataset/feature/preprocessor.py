import polars as pl
import os
from typing import Dict

from dataset.datainfo import RawInfo, RawReader, DATA_PATH
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe
from dataset.const import TOPICS


class Preprocessor:
    RAW_INFO = RawInfo()
    DEPTH_2_TO_1_QUERY: Dict[str, str] = {
        'applprev': """
            SELECT case_id, num_group1
                , count(case when (cacccardblochreas_147M = 'P33_145_161') then 1 else null end) as count__if__cacccardblochreas_147m_eq_p33_145_161__then_1__L
                , count(case when (cacccardblochreas_147M = 'P201_63_60') then 1 else null end) as count__if__cacccardblochreas_147m_eq_p201_63_60__then_1__L
                , count(case when (conts_type_509L = 'PRIMARY_MOBILE') then 1 else null end) as count__if__conts_type_509l_eq_primary_mobile__then_1__L
                , count(case when (conts_type_509L = 'HOME_PHONE') then 1 else null end) as count__if__conts_type_509l_eq_home_phone__then_1__L
                , count(case when (conts_type_509L = 'EMPLOYMENT_PHONE') then 1 else null end) as count__if__conts_type_509l_eq_employment_phone__then_1__L
                , count(case when (credacc_cards_status_52L = 'CANCELLED') then 1 else null end) as count__if__credacc_cards_status_52l_eq_cancelled__then_1__L
                , count(case when (credacc_cards_status_52L = 'ACTIVE') then 1 else null end) as count__if__credacc_cards_status_52l_eq_active__then_1__L
                , count(case when (credacc_cards_status_52L = 'INACTIVE') then 1 else null end) as count__if__credacc_cards_status_52l_eq_inactive__then_1__L
                , count(case when (credacc_cards_status_52L = 'BLOCKED') then 1 else null end) as count__if__credacc_cards_status_52l_eq_blocked__then_1__L
                , count(case when (credacc_cards_status_52L = 'RENEWED') then 1 else null end) as count__if__credacc_cards_status_52l_eq_renewed__then_1__L
                , count(case when (credacc_cards_status_52L = 'UNCONFIRMED') then 1 else null end) as count__if__credacc_cards_status_52l_eq_unconfirmed__then_1__L
                , count(distinct cacccardblochreas_147M) as count_distinct_cacccardblochreas_147m__L
                , count(distinct conts_type_509L) as count_distinct_conts_type_509l__L
                , count(distinct credacc_cards_status_52L) as count_distinct_credacc_cards_status_52l__L
            from data
            group by case_id, num_group1
            """,
        'person': """
            SELECT case_id, num_group1
                , count(case when (addres_district_368M = 'P125_48_164') then 1 else null end) as count__if__addres_district_368m_eq_p125_48_164__then_1__L
                , count(case when (addres_district_368M = 'P155_139_77') then 1 else null end) as count__if__addres_district_368m_eq_p155_139_77__then_1__L
                , count(case when (addres_district_368M = 'P114_74_190') then 1 else null end) as count__if__addres_district_368m_eq_p114_74_190__then_1__L
                , count(case when (addres_role_871L = 'PERMANENT') then 1 else null end) as count__if__addres_role_871l_eq_permanent__then_1__L
                , count(case when (addres_role_871L = 'CONTACT') then 1 else null end) as count__if__addres_role_871l_eq_contact__then_1__L
                , count(case when (addres_role_871L = 'TEMPORARY') then 1 else null end) as count__if__addres_role_871l_eq_temporary__then_1__L
                , count(case when (addres_role_871L = 'REGISTERED') then 1 else null end) as count__if__addres_role_871l_eq_registered__then_1__L
                , count(case when (addres_zip_823M = 'P161_14_174') then 1 else null end) as count__if__addres_zip_823m_eq_p161_14_174__then_1__L
                , count(case when (addres_zip_823M = 'P144_138_111') then 1 else null end) as count__if__addres_zip_823m_eq_p144_138_111__then_1__L
                , count(case when (addres_zip_823M = 'P46_103_143') then 1 else null end) as count__if__addres_zip_823m_eq_p46_103_143__then_1__L
                , count(case when (conts_role_79M = 'P38_92_157') then 1 else null end) as count__if__conts_role_79m_eq_p38_92_157__then_1__L
                , count(case when (conts_role_79M = 'P177_137_98') then 1 else null end) as count__if__conts_role_79m_eq_p177_137_98__then_1__L
                , count(case when (conts_role_79M = 'P7_147_157') then 1 else null end) as count__if__conts_role_79m_eq_p7_147_157__then_1__L
                , count(case when (empls_economicalst_849M = 'P22_131_138') then 1 else null end) as count__if__empls_economicalst_849m_eq_p22_131_138__then_1__L
                , count(case when (empls_economicalst_849M = 'P164_110_33') then 1 else null end) as count__if__empls_economicalst_849m_eq_p164_110_33__then_1__L
                , count(case when (empls_economicalst_849M = 'P28_32_178') then 1 else null end) as count__if__empls_economicalst_849m_eq_p28_32_178__then_1__L
                , count(case when (empls_economicalst_849M = 'P148_57_109') then 1 else null end) as count__if__empls_economicalst_849m_eq_p148_57_109__then_1__L
                , count(case when (empls_economicalst_849M = 'P112_86_147') then 1 else null end) as count__if__empls_economicalst_849m_eq_p112_86_147__then_1__L
                , count(case when (empls_economicalst_849M = 'P191_80_124') then 1 else null end) as count__if__empls_economicalst_849m_eq_p191_80_124__then_1__L
                , count(case when (empls_economicalst_849M = 'P7_47_145') then 1 else null end) as count__if__empls_economicalst_849m_eq_p7_47_145__then_1__L
                , count(case when (empls_economicalst_849M = 'P164_122_65') then 1 else null end) as count__if__empls_economicalst_849m_eq_p164_122_65__then_1__L
                , count(case when (empls_economicalst_849M = 'P82_144_169') then 1 else null end) as count__if__empls_economicalst_849m_eq_p82_144_169__then_1__L
                , count(case when (empls_employer_name_740M = 'P114_118_163') then 1 else null end) as count__if__empls_employer_name_740m_eq_p114_118_163__then_1__L
                , count(case when (empls_employer_name_740M = 'P179_55_175') then 1 else null end) as count__if__empls_employer_name_740m_eq_p179_55_175__then_1__L
                , count(case when (empls_employer_name_740M = 'P26_112_122') then 1 else null end) as count__if__empls_employer_name_740m_eq_p26_112_122__then_1__L
                , count(distinct addres_district_368M) as count_addres_district_368m__L
                , count(distinct addres_role_871L) as count_distinct_addres_role_871l__L
                , count(distinct addres_zip_823M) as count_distinct_addres_zip_823m__L
                , count(distinct conts_role_79M) as count_distinct_conts_role_79m__L
                , count(distinct empls_economicalst_849M) as count_distinct_empls_economicalst_849m__L
                , count(distinct empls_employedfrom_796D) as count_distinct_empls_employedfrom_796d__L
                , min(empls_employedfrom_796D) as min_empls_employedfrom_796d__D
                , max(empls_employedfrom_796D) as max_empls_employedfrom_796d__D
                , count(distinct empls_employer_name_740M) as count_distinct_empls_employer_name_740m__L
                , count(distinct relatedpersons_role_762T) as count_distinct_relatedpersons_role_762t__L
            from data
            group by case_id, num_group1
            """,
        'credit_bureau_b': """
            SELECT case_id, num_group1
                , count(num_group2) as count__num_group2__L
                , max(pmts_date_1107D) as max_pmts_date_1107d_D
                , min(pmts_date_1107D) as min_pmts_date_1107d_D
                , sum(pmts_dpdvalue_108P) as sum_pmts_dpdvalue_108p_P
                , sum(pmts_pmtsoverdue_635A) as sum_pmts_pmtsoverdue_635a_A
            from data
            group by case_id, num_group1
            """,
        'credit_bureau_a': """
            SELECT case_id, num_group1
                , count(status) as count_status__L
                , sum(status) as sum_status__L
                , avg(status) as avg_status__L
                , count(distinct collater_typofvalofguarant_298M407M) as count_distinct_collater_typofvalofguarant_298M407M__L
                , sum(collater_valueofguarantee_1124L876L) as sum_collater_valueofguarantee_1124L876L__L
                , count(distinct collaterals_typeofguarante_359M669M) as count_distinct_collaterals_typeofguarante_359M669M__L
                , sum(pmts_dpd_1073P303P) as sum_pmts_dpd_1073P303P__P
                , sum(pmts_month_158T706T) as sum_pmts_month_158T706T__T
                , sum(pmts_overdue_1140A1152A) as sum_pmts_overdue_1140A1152A__A
                , min(concat(pmts_year_1139T507T, '-01-01')) as min_pmts_year_1139T507T__D
                , max(concat(pmts_year_1139T507T, '-01-01')) as max_pmts_year_1139T507T__D
                , count(distinct subjectroles_name_541M838M) as count_distinct_subjectroles_name_541M838M__L
                --, count(distinct case when status = 1 then collater_typofvalofguarant_298M407M else null end) as count_distinct__if__status_eq_1_then_collater_typofvalofguarant_298m407m__L
                --, sum(case when status = 1 then collater_valueofguarantee_1124L876L else null end) as sum__if__status_eq_1_then_collater_valueofguarantee_1124l876l__L
                --, count(distinct case when status = 1 then collaterals_typeofguarante_359M669M else null end) as count_distinct__if__status_eq_1_then_collaterals_typeofguarante_359m669m__L
                --, sum(case when status = 1 then pmts_dpd_1073P303P else null end) as sum__if__status_eq_1_then_pmts_dpd_1073p303p__P
                --, sum(case when status = 1 then pmts_month_158T706T else null end) as sum__if__status_eq_1_then_pmts_month_158t706t__T
                --, sum(case when status = 1 then pmts_overdue_1140A1152A else null end) as sum__if__status_eq_1_then_pmts_overdue_1140a1152a__A
                --, min(case when status = 1 then concat(pmts_year_1139T507T, '-01-01') else null end) as min__if__status_eq_1_then_pmts_year_1139t507t__D
                --, max(case when status = 1 then concat(pmts_year_1139T507T, '-01-01') else null end) as max__if__status_eq_1_then_pmts_year_1139t507t__D
                --, count(distinct case when status = 1 then subjectroles_name_541M838M else null end) as count_distinct__if__status_eq_1_then_subjectroles_name_541m838m__L
                --, count(distinct case when status = 0 then collater_typofvalofguarant_298M407M else null end) as count_distinct__if__status_eq_0_then_collater_typofvalofguarant_298m407m__L
                --, sum(case when status = 0 then collater_valueofguarantee_1124L876L else null end) as sum__if__status_eq_0_then_collater_valueofguarantee_1124l876l__L
                --, count(distinct case when status = 0 then collaterals_typeofguarante_359M669M else null end) as count_distinct__if__status_eq_0_then_collaterals_typeofguarante_359m669m__L
                --, sum(case when status = 0 then pmts_dpd_1073P303P else null end) as sum__if__status_eq_0_then_pmts_dpd_1073p303p__P
                --, sum(case when status = 0 then pmts_month_158T706T else null end) as sum__if__status_eq_0_then_pmts_month_158t706t__T
                --, sum(case when status = 0 then pmts_overdue_1140A1152A else null end) as sum__if__status_eq_0_then_pmts_overdue_1140a1152a__A
                --, min(case when status = 0 then concat(pmts_year_1139T507T, '-01-01') else null end) as min__if__status_eq_0_then_pmts_year_1139t507t__D
                --, max(case when status = 0 then concat(pmts_year_1139T507T, '-01-01') else null end) as max__if__status_eq_0_then_pmts_year_1139t507t__D
                --, count(distinct case when status = 0 then subjectroles_name_541M838M else null end) as count_distinct__if__status_eq_0_then_subjectroles_name_541m838m__L
                , count(case when (collater_typofvalofguarant_298M407M = '9a0c095e') then 1 else null end) as count__if__collater_typofvalofguarant_298m_eq_9a0c095e__then_1__L
                , count(case when (collater_typofvalofguarant_298M407M = '8fd95e4b') then 1 else null end) as count__if__collater_typofvalofguarant_298m_eq_8fd95e4b__then_1__L
                , count(case when (collaterals_typeofguarante_359M669M = 'c7a5ad39') then 1 else null end) as count__if__collaterals_typeofguarante_359m_eq_c7a5ad39__then_1__L
                , count(case when (collaterals_typeofguarante_359M669M = '3cbe86ba') then 1 else null end) as count__if__collaterals_typeofguarante_359m_eq_3cbe86ba__then_1__L
                , count(case when (subjectroles_name_541M838M = 'ab3c25cf') then 1 else null end) as count__if__subjectroles_name_541m_eq_ab3c25cf__then_1__L
                , count(case when (subjectroles_name_541M838M = '15f04f45') then 1 else null end) as count__if__subjectroles_name_541m_eq_15f04f45__then_1__L
            from data
            group by case_id, num_group1
            """,
    }

    def __init__(self, type_: str):
        self.type_ = type_

    def preprocess(self):
        for topic in TOPICS:
            print(f'\n[*] Preprocessing {topic.name}, depth={topic.depth}')
            if topic.depth <= 1 and topic.name not in self.DEPTH_2_TO_1_QUERY:
                print(f'  [+] Memory optimization {topic.name}')
                self._memory_opt(topic.name, depth=topic.depth)
            elif topic.depth <= 1 and topic.name in self.DEPTH_2_TO_1_QUERY:
                print(f'  [+] skip {topic.name} because it is in DEPTH_2_TO_1_QUERY')
            elif topic.depth == 2 and topic.name in self.DEPTH_2_TO_1_QUERY:
                query = self.DEPTH_2_TO_1_QUERY[topic.name]
                if topic.name == 'credit_bureau_a':
                    self._preprocess_cb_a(topic.name, query)
                    print(f'  [+++] skip {topic.name} because it is credit_bureau_a')
                else:
                    self._preprocess_each(topic.name, query)
            elif topic.depth == 2 and topic.name not in self.DEPTH_2_TO_1_QUERY:
                raise ValueError(f'No query for {topic.name} in DEPTH_2_TO_1_QUERY but it is depth=2 topic')

    def _memory_opt(self, topic: str, depth: int):
        data = self.RAW_INFO.read_raw(topic, depth=depth, reader=RawReader('polars'), type_=self.type_)
        data = optimize_dataframe(data, verbose=True)
        self.RAW_INFO.save_as_prep(data, topic, depth=depth, type_=self.type_)

    def _join_depth2_0(self, depth1, depth2):
        depth2 = depth2.filter(pl.col('num_group2') == 0).drop('num_group2')
        depth1 = depth1.join(depth2, on=['case_id', 'num_group1'], how='left')
        return depth1

    def _preprocess_each(self, topic: str, query: str):
        depth2 = self.RAW_INFO.read_raw(topic, depth=2, reader=RawReader('polars'), type_=self.type_)
        depth1 = self.RAW_INFO.read_raw(topic, depth=1, reader=RawReader('polars'), type_=self.type_)
        temp = pl.SQLContext(data=depth2).execute(query, eager=True)
        depth1 = depth1.join(temp, on=['case_id', 'num_group1'], how='left')
        depth1 = self._join_depth2_0(depth1, depth2)

        depth1 = optimize_dataframe(depth1, verbose=True)
        self.RAW_INFO.save_as_prep(depth1, topic, depth=1, type_=self.type_)

    def _preprocess_cb_a(self, topic: str, query: str):
        depth2 = self.RAW_INFO.read_raw(topic, depth=2, reader=RawReader('polars'), type_=self.type_)
        print('[*] Read depth=2 data')
        depth2 = optimize_dataframe(depth2, verbose=True)

        depth1 = self.RAW_INFO.read_raw(topic, depth=1, reader=RawReader('polars'), type_=self.type_)
        print('[*] Read depth=1 data')
        depth1 = optimize_dataframe(depth1, verbose=True)
        depth1 = self._join_depth2_0(depth1, depth2)

        temp_path = (
            DATA_PATH / 'parquet_preps' / self.type_ / f"{self.type_}_{topic}_temp.parquet"
        )
        print(f'[*] Temp path: {temp_path}')
        if not os.path.exists(temp_path):
            depth2_temp = pl.SQLContext(data=depth2).execute(
                '''
                SELECT case_id, num_group1, 1 as status
                    , collater_typofvalofguarant_298M as collater_typofvalofguarant_298M407M
                    , collater_valueofguarantee_1124L as collater_valueofguarantee_1124L876L
                    , collaterals_typeofguarante_359M as collaterals_typeofguarante_359M669M
                    , pmts_dpd_1073P as pmts_dpd_1073P303P
                    , pmts_month_158T as pmts_month_158T706T
                    , pmts_overdue_1140A as pmts_overdue_1140A1152A
                    , pmts_year_1139T as pmts_year_1139T507T
                    , subjectroles_name_541M as subjectroles_name_541M838M
                from data
                union all
                SELECT case_id, num_group1, 0 as status
                    , collater_typofvalofguarant_407M as collater_typofvalofguarant_298M407M
                    , collater_valueofguarantee_876L as collater_valueofguarantee_1124L876L
                    , collaterals_typeofguarante_669M as collaterals_typeofguarante_359M669M
                    , pmts_dpd_303P as pmts_dpd_1073P303P
                    , pmts_month_706T as pmts_month_158T706T
                    , pmts_overdue_1152A as pmts_overdue_1140A1152A
                    , pmts_year_507T as pmts_year_1139T507T
                    , subjectroles_name_838M as subjectroles_name_541M838M
                from data
                ''',
                eager=True,
            )
            depth2_temp.write_parquet(temp_path)
            print('[*] Temp data saved')
        else:
            depth2_temp = pl.read_parquet(temp_path)
            print('[*] Read temp data')
        depth2_temp = pl.SQLContext(data=depth2_temp).execute(query, eager=True)
        depth2_temp = optimize_dataframe(depth2_temp, verbose=True)
        depth1 = depth1.join(depth2_temp, on=['case_id', 'num_group1'], how='left')

        self.RAW_INFO.save_as_prep(depth1, topic, depth=1, type_=self.type_)

if __name__ == "__main__":
    prep = Preprocessor('train')
    prep.preprocess()


# fd = FeatureDefiner('person', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')

# fd = FeatureDefiner('applprev', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')

# fd = FeatureDefiner('credit_bureau_a', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')

# fd = FeatureDefiner('credit_bureau_b', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')
