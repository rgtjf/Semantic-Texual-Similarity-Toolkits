# coding: utf8
from __future__ import print_function

import os
import time
import traceback

import lxml.html
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait

from stst import utils
from stst.modules.features import Feature
from stst import config


class AsiyaDriver(object):
    def __init__(self):
        'stst\resources\linux\chromedriver'
        # cur_dir = os.path.dirname(__file__) # 'stst/features'
        # path = os.path.join(cur_dir, 'resources')
        # print(cur_dir)
        driver = webdriver.Chrome()
        driver.get("http://asiya.cs.upc.edu/demo/asiya_online.php#")
        time.sleep(3)
        driver._switch_to.frame(driver.find_element_by_id("demo-content"))
        elem = Select(driver.find_element_by_id("input"))
        elem.select_by_value("raw")
        elem = driver.find_element_by_id("no_tok")
        elem.click()
        elem = Select(driver.find_element_by_id("srclang"))
        elem.select_by_value("en")
        elem = Select(driver.find_element_by_id("trglang"))
        elem.select_by_value("en")
        elem = Select(driver.find_element_by_id("srccase"))
        elem.select_by_value("ci")
        elem = Select(driver.find_element_by_id("trgcase"))
        elem.select_by_value("ci")

        self.driver = driver

    def reload(self):
        driver = self.driver
        driver.get("http://asiya.cs.upc.edu/demo/asiya_online.php#")
        time.sleep(3)
        driver._switch_to.frame(driver.find_element_by_id("demo-content"))
        elem = Select(driver.find_element_by_id("input"))
        elem.select_by_value("raw")
        elem = driver.find_element_by_id("no_tok")
        elem.click()
        elem = Select(driver.find_element_by_id("srclang"))
        elem.select_by_value("en")
        elem = Select(driver.find_element_by_id("trglang"))
        elem.select_by_value("en")
        elem = Select(driver.find_element_by_id("srccase"))
        elem.select_by_value("ci")
        elem = Select(driver.find_element_by_id("trgcase"))
        elem.select_by_value("ci")
        self.driver = driver

    def run(self, sa, sb):
        driver = self.driver
        try:

            elem = driver.find_element_by_id("srctext")
            elem.send_keys(sa)

            elem = driver.find_element_by_id("systext")
            elem.send_keys(sb)

            elem = driver.find_element_by_id("reftext")
            elem.send_keys(sb)

            elem = driver.find_element_by_id("submitbtndoasiya")
            elem.click()

            elem = WebDriverWait(driver, 300).until(
                EC.presence_of_element_located((By.ID, "div_scores"))
            )
            headers = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//thead//tr//th")

            rows = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//tbody//tr//td")

            rows = [eval(row.text) for row in rows[3:]]

            # BLEU, GTM - 3, NIST, -WER, -PER, Ol, -TERbase, METEOR - ex, ROUGE - L

            elem = driver.find_element_by_xpath("//input[@value='Clear Files']")
            elem.click()
        except:
            traceback.print_exc()
            rows = []
        return rows

    def run_file(self, src='sa.txt', sys='sb.txt'):
        ''' write to file sa.txt, sb.txt'''
        cur_dir = os.path.dirname(__file__)  # 'stst/features'
        # path = os.path.join(cur_dir, 'resources')
        print(cur_dir)

        driver = self.driver
        wait = 3
        try:
            elem = driver.find_element_by_id("srcupload")
            elem.click()
            time.sleep(wait)
            cmd = r'stst\features\upload.exe '
            os.system(cmd + src)
            time.sleep(wait)

            elem = driver.find_element_by_id("sysupload")
            elem.click()
            time.sleep(wait)

            os.system(cmd + sys)
            time.sleep(wait)

            elem = driver.find_element_by_id("refupload")
            elem.click()
            time.sleep(wait)
            os.system(cmd + src)
            time.sleep(wait)

            elem = driver.find_element_by_id("submitbtndoasiya")
            elem.click()

            elem = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.ID, "div_scores"))
            )

            page = driver.page_source
            # headers = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//thead//tr//th")

            # rows = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//tbody//tr//td")
            # rows = [eval(row.text) for row in rows[3:]]

            # BLEU, GTM - 3, NIST, -WER, -PER, Ol, -TERbase, METEOR - ex, ROUGE - L

            elem = driver.find_element_by_xpath("//input[@value='Clear Files']")
            elem.click()
        except:
            traceback.print_exc()
            # rows = []
            page = ''
        return page

    def extract_table(self, page_source):
        root = lxml.html.fromstring(page_source)
        features = []
        for row in root.xpath('.//table[@id="tab_scores_mmatrix_grseg"]//tbody//tr'):
            cells = row.xpath('.//td/text()')
            feature = [eval(cell.strip()) for cell in cells[2:]]
            features.append(feature)
        return features

    def quit(self):
        self.driver.close()


class AsiyaMTFeature(Feature):
    def extract_instances(self, train_instances):
        asiya = AsiyaDriver()

        n_lines = 250
        features = []
        infos = []

        idx_list = range(0, len(train_instances), n_lines)

        for idx in idx_list:
            st, ed = idx, idx + n_lines
            if ed > len(train_instances):
                ed = len(train_instances)
            print("\rAsiya MT Featyre index = %d, st = %d, ed = %d" % (idx, st, ed), end=' ')

            while True:
                ''' sa -> sb '''
                f_sa = utils.create_write_file(config.TMP_DIR + '/sa.txt')
                f_sb = utils.create_write_file(config.TMP_DIR + '/sb.txt')
                for id in range(st, ed):
                    lemma_sa, lemma_sb = train_instances[id].get_word(type='lemma')
                    lemma_sa = ' '.join(lemma_sa)
                    lemma_sb = ' '.join(lemma_sb)
                    print(lemma_sa, file=f_sa)
                    print(lemma_sb, file=f_sb)
                f_sa.close()
                f_sb.close()
                page = asiya.run_file()
                if page != ' ':
                    features_sa = asiya.extract_table(page)
                    break
                else:
                    asiya.reload()

            while True:
                ''' sb -> sa '''
                f_sa = utils.create_write_file(config.TMP_DIR + '/sb.txt')
                f_sb = utils.create_write_file(config.TMP_DIR + '/sa.txt')
                # "F:\PyCharmWorkSpace\SemEval17_T1_System\resources\external_dict\sa.txt"
                for id in range(st, ed):
                    lemma_sa, lemma_sb = train_instances[id].get_word(type='lemma')
                    lemma_sa = ' '.join(lemma_sa)
                    lemma_sb = ' '.join(lemma_sb)
                    print(lemma_sa, file=f_sa)
                    print(lemma_sb, file=f_sb)
                f_sa.close()
                f_sb.close()
                page = asiya.run_file()
                if page != ' ':
                    features_sb = asiya.extract_table(page)
                    break
                else:
                    asiya.reload()

            ''' Merge feature '''
            for a, b in zip(features_sa, features_sb):
                features.append(a + b)
                infos.append([])
        print(features[:10])
        return features, infos


class AsiyaEsEsMTFeature(Feature):
    def extract_instances(self, train_instances):
        asiya = AsiyaDriver()

        n_lines = 250
        features = []
        infos = []

        idx_list = range(0, len(train_instances), n_lines)

        for idx in idx_list:
            st, ed = idx, idx + n_lines
            if ed > len(train_instances):
                ed = len(train_instances)
            print("\rAsiya MT Featyre index = %d, st = %d, ed = %d" % (idx, st, ed), end=' ')

            while True:
                ''' sa -> sb '''
                f_sa = utils.create_write_file(config.EX_DICT_DIR + '/sa.txt')
                f_sb = utils.create_write_file(config.EX_DICT_DIR + '/sb.txt')
                for id in range(st, ed):
                    lemma_sa, lemma_sb = train_instances[id].get_word(type='lemma')
                    lemma_sa = ' '.join(lemma_sa)
                    lemma_sb = ' '.join(lemma_sb)
                    print(lemma_sa, file=f_sa)
                    print(lemma_sb, file=f_sb)
                f_sa.close()
                f_sb.close()
                page = asiya.run_file()
                if page != ' ':
                    features_sa = asiya.extract_table(page)
                    break
                else:
                    asiya.reload()
            features += features_sa
            infos += [['0']] * len(features)
        print(features[:10])
        return features, infos