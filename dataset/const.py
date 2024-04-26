from dataclasses import dataclass
from typing import Dict

KEY_COL = ["case_id"]
DATE_COL = ["date_decision"]
TARGET_COL = ["target"]

TOPIC_NAMES = [
    "applprev",
    "credit_bureau_a",
    "credit_bureau_b",
    "person",
    "debitcard",
    "deposit",
    "other",
    "tax_registry_a",
    "tax_registry_b",
    "tax_registry_c",
    "static",
    "static_cb",
]

@dataclass
class Topic:
    name: str
    depth: int

    def __post_init__(self):
        if self.name not in TOPIC_NAMES:
            raise ValueError(f"Invalid topic: {self.name}. Available topics: {TOPIC_NAMES}")

TOPICS = [
    Topic("applprev", 1),
    Topic("applprev", 2),
    Topic("credit_bureau_a", 1),
    Topic("credit_bureau_a", 2),
    Topic("credit_bureau_b", 1),
    Topic("credit_bureau_b", 2),
    Topic("person", 1),
    Topic("person", 2),
    Topic("debitcard", 1),
    Topic("deposit", 1),
    Topic("other", 1),
    Topic("tax_registry_a", 1),
    Topic("tax_registry_b", 1),
    Topic("tax_registry_c", 1),
    Topic("static", 0),
    Topic("static_cb", 0),
]

QUERY_FORMAT_REPLACEMENTS = [
    ('<', 'lt'),
    ('<=', 'le'),
    ('>', 'gt'),
    ('>=', 'ge'),
    ('!=', 'ne'),
    ('=', 'eq'),
    ('-', 'sub'),
    ('+', 'plus'),
    ('/', 'div'),
    ('*', 'mul'),
    ('case when ', '_if_'),
    ('else null end', ''),
    ('(', '_'),
    (')', '_'),
    ("'", ""),
    (' and ', ''),
    (' else ', '_'),
    (' ', '_'),
]

CB_A_PREPREP_QUERY = """
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
    """

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
                , min(case when concat(pmts_year_1139T507T, '-01-01')='-01-01' then null else replace(concat(pmts_year_1139T507T, '-01-01'), '.0', '') end) as min_pmts_year_1139T507T__D
                , max(case when concat(pmts_year_1139T507T, '-01-01')='-01-01' then null else replace(concat(pmts_year_1139T507T, '-01-01'), '.0', '') end) as max_pmts_year_1139T507T__D
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
