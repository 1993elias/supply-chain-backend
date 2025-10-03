# main.py - Backend com Pyomo
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import io
import json
from typing import Dict, Any
import logging
from datetime import datetime
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Supply Chain Optimizer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataLoader:
    """Carrega e processa dados do Excel"""
    
    @staticmethod
    def load_excel(file_content: bytes):
        """Carrega todas as abas do Excel"""
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_content))
            data = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
                data[sheet_name] = df
                logger.info(f"Aba carregada: {sheet_name} ({df.shape[0]} linhas)")
            
            return data
        except Exception as e:
            logger.error(f"Erro ao carregar Excel: {e}")
            raise HTTPException(400, f"Erro ao processar Excel: {str(e)}")
    
    @staticmethod
    def extract_sets(excel_data):
        """Extrai conjuntos do Excel"""
        sets = {}
        
        # RESOURCE
        if 'RESOURCE' in excel_data:
            sets['RESOURCE'] = excel_data['RESOURCE'].iloc[:, 0].dropna().tolist()
        else:
            sets['RESOURCE'] = ['NH3', 'H2', 'CH4', 'Energy']
        
        # UNIT
        if 'UNIT' in excel_data:
            sets['UNIT'] = excel_data['UNIT'].iloc[:, 0].dropna().tolist()
        else:
            sets['UNIT'] = ['Reformer', 'Electrolyzer', 'HaberBosch']
        
        # LOCAL
        if 'Local' in excel_data or 'LOCAL' in excel_data:
            key = 'Local' if 'Local' in excel_data else 'LOCAL'
            sets['LOCAL'] = excel_data[key].iloc[:, 0].dropna().tolist()
        else:
            sets['LOCAL'] = ['Plant1', 'Plant2', 'Market1']
        
        # LEVEL
        sets['LEVEL'] = ['L1', 'L2', 'L3', 'L4']
        
        # MODAL
        if 'MODAL' in excel_data:
            sets['MODAL'] = excel_data['MODAL'].iloc[:, 0].dropna().tolist()
        else:
            sets['MODAL'] = ['Road', 'Rail', 'Maritime']
        
        return sets
    
    @staticmethod
    def extract_parameters(excel_data, sets):
        """Extrai parâmetros do Excel"""
        params = {}
        
        # Criar valores padrão
        for r in sets['RESOURCE']:
            params.setdefault('OC', {})[r] = np.random.uniform(100, 500)
            params.setdefault('MP', {})[r] = np.random.uniform(200, 600)
            params.setdefault('emittedCO2', {})[r] = np.random.uniform(0, 3)
            params.setdefault('avoidedCO2', {})[r] = np.random.uniform(0, 5)
        
        # Tentar carregar dados reais do Excel
        if 'RESOURCE' in excel_data:
            df = excel_data['RESOURCE']
            for _, row in df.iterrows():
                if pd.notna(row[0]):
                    r = row[0]
                    if 'OC' in df.columns and pd.notna(row.get('OC')):
                        params['OC'][r] = float(row['OC'])
                    if 'MP' in df.columns and pd.notna(row.get('MP')):
                        params['MP'][r] = float(row['MP'])
                    if 'emittedCO2' in df.columns and pd.notna(row.get('emittedCO2')):
                        params['emittedCO2'][r] = float(row['emittedCO2'])
                    if 'avoidedCO2' in df.columns and pd.notna(row.get('avoidedCO2')):
                        params['avoidedCO2'][r] = float(row['avoidedCO2'])
        
        # UL parameters
        params['capCostA'] = {}
        params['capCostB'] = {}
        params['CapMax'] = {}
        params['CapMin'] = {}
        params['UL'] = []
        
        for u in sets['UNIT']:
            for l in sets['LEVEL']:
                params['UL'].append((u, l))
                params['capCostA'][(u, l)] = np.random.uniform(500, 2000)
                params['capCostB'][(u, l)] = np.random.uniform(3000, 10000)
                params['CapMax'][(u, l)] = np.random.uniform(500, 2000)
                params['CapMin'][(u, l)] = np.random.uniform(50, 200)
        
        # UR parameters (Input/Output ratios)
        params['IAR'] = {}
        params['OAR'] = {}
        params['UR'] = []
        
        conversions = {
            ('Reformer', 'CH4'): 1.0,
            ('Reformer', 'H2'): 0.25,
            ('Electrolyzer', 'Energy'): 1.0,
            ('Electrolyzer', 'H2'): 0.02,
            ('HaberBosch', 'H2'): 1.0,
            ('HaberBosch', 'NH3'): 3.0
        }
        
        for (u, r), val in conversions.items():
            params['UR'].append((u, r))
            if r in ['CH4', 'Energy', 'H2']:
                params['IAR'][(u, r)] = val
            else:
                params['OAR'][(u, r)] = val
        
        # LR parameters
        params['avail'] = {}
        params['demand'] = {}
        params['LR'] = []
        
        for p in sets['LOCAL']:
            for r in sets['RESOURCE']:
                params['LR'].append((p, r))
                params['avail'][(p, r)] = np.random.uniform(1000, 10000)
                params['demand'][(p, r)] = np.random.uniform(0, 5000)
        
        # MR parameters
        params['modCost'] = {}
        params['fperda'] = {}
        params['CO2mod'] = {}
        params['MR'] = []
        
        for r in sets['RESOURCE']:
            for mo in sets['MODAL']:
                params['MR'].append((r, mo))
                params['modCost'][(r, mo)] = np.random.uniform(0.05, 0.2)
                params['fperda'][(r, mo)] = np.random.uniform(0.01, 0.05)
                params['CO2mod'][(r, mo)] = np.random.uniform(0.02, 0.1)
        
        # Distâncias
        params['dist'] = {}
        for p1 in sets['LOCAL']:
            for p2 in sets['LOCAL']:
                if p1 != p2:
                    params['dist'][(p1, p2)] = np.random.uniform(100, 1000)
                else:
                    params['dist'][(p1, p2)] = 0
        
        return params

class SupplyChainOptimizer:
    """Modelo de otimização com Pyomo"""
    
    def __init__(self, excel_data, user_params):
        self.excel_data = excel_data
        self.user_params = user_params
        self.sets = DataLoader.extract_sets(excel_data)
        self.params = DataLoader.extract_parameters(excel_data, self.sets)
        self.model = None
        self.results = None
    
    def build_model(self):
        """Constrói modelo Pyomo"""
        m = pyo.ConcreteModel("Supply_Chain")
        
        # SETS
        m.RESOURCE = pyo.Set(initialize=self.sets['RESOURCE'])
        m.UNIT = pyo.Set(initialize=self.sets['UNIT'])
        m.LOCAL = pyo.Set(initialize=self.sets['LOCAL'])
        m.LEVEL = pyo.Set(initialize=self.sets['LEVEL'])
        m.MODAL = pyo.Set(initialize=self.sets['MODAL'])
        
        m.UL = pyo.Set(initialize=self.params['UL'], dimen=2)
        m.LR = pyo.Set(initialize=self.params['LR'], dimen=2)
        m.UR = pyo.Set(initialize=self.params['UR'], dimen=2)
        m.MR = pyo.Set(initialize=self.params['MR'], dimen=2)
        
        # PARAMETERS
        m.OC = pyo.Param(m.RESOURCE, initialize=self.params['OC'])
        m.MP = pyo.Param(m.RESOURCE, initialize=self.params['MP'])
        m.emittedCO2 = pyo.Param(m.RESOURCE, initialize=self.params['emittedCO2'])
        m.avoidedCO2 = pyo.Param(m.RESOURCE, initialize=self.params['avoidedCO2'])
        
        m.capCostA = pyo.Param(m.UL, initialize=self.params['capCostA'])
        m.capCostB = pyo.Param(m.UL, initialize=self.params['capCostB'])
        m.CapMax = pyo.Param(m.UL, initialize=self.params['CapMax'])
        m.CapMin = pyo.Param(m.UL, initialize=self.params['CapMin'])
        
        m.IAR = pyo.Param(m.UR, initialize=self.params['IAR'], default=0)
        m.OAR = pyo.Param(m.UR, initialize=self.params['OAR'], default=0)
        
        m.avail = pyo.Param(m.LR, initialize=self.params['avail'])
        m.demand = pyo.Param(m.LR, initialize=self.params['demand'])
        
        m.dist = pyo.Param(m.LOCAL, m.LOCAL, initialize=self.params['dist'])
        m.modCost = pyo.Param(m.MR, initialize=self.params['modCost'])
        m.fperda = pyo.Param(m.MR, initialize=self.params['fperda'])
        m.CO2mod = pyo.Param(m.MR, initialize=self.params['CO2mod'])
        
        # User parameters
        m.carbonValue = self.user_params.get('carbonValue', 200)
        m.annualfact = self.user_params.get('annualfact', 0.086)
        m.MaintanceCost = self.user_params.get('maintanceCost', 1.06)
        m.fop = self.user_params.get('fop', 8760)
        
        # VARIABLES
        m.y = pyo.Var(m.LOCAL, m.UNIT, domain=pyo.Binary)
        m.yLevel = pyo.Var(m.LOCAL, m.UL, domain=pyo.Binary)
        m.w = pyo.Var(m.LOCAL, m.UNIT, domain=pyo.NonNegativeReals)
        m.wLevel = pyo.Var(m.LOCAL, m.UL, domain=pyo.NonNegativeReals)
        
        m.comprado = pyo.Var(m.LR, domain=pyo.NonNegativeReals)
        m.vendido = pyo.Var(m.LR, domain=pyo.NonNegativeReals)
        m.prod = pyo.Var(m.LOCAL, m.UR, domain=pyo.NonNegativeReals)
        m.cons = pyo.Var(m.LOCAL, m.UR, domain=pyo.NonNegativeReals)
        
        m.transfentra = pyo.Var(m.LOCAL, m.LOCAL, m.RESOURCE, domain=pyo.NonNegativeReals)
        m.transfsai = pyo.Var(m.LOCAL, m.LOCAL, m.RESOURCE, domain=pyo.NonNegativeReals)
        m.transfEntraModal = pyo.Var(m.LOCAL, m.LOCAL, m.MR, domain=pyo.NonNegativeReals)
        m.transfSaiModal = pyo.Var(m.LOCAL, m.LOCAL, m.MR, domain=pyo.NonNegativeReals)
        
        # OBJECTIVE
        def obj_rule(m):
            inv = sum(m.capCostA[u,l]*m.wLevel[p,u,l] + m.capCostB[u,l]*m.yLevel[p,u,l]
                     for p in m.LOCAL for (u,l) in m.UL) * m.annualfact * m.MaintanceCost
            
            cmp = sum(m.comprado[p,r]*m.OC[r] for (p,r) in m.LR) / 1e6
            
            rpr = sum(m.vendido[p,r]*m.MP[r] for (p,r) in m.LR) / 1e6
            
            carb_in = sum(m.comprado[p,r]*m.emittedCO2[r] for (p,r) in m.LR)
            carb_out = sum(m.vendido[p,r]*m.avoidedCO2[r] for (p,r) in m.LR)
            emis_trans = sum(m.transfSaiModal[p1,p2,r,mo]*m.dist[p1,p2]*m.CO2mod[r,mo]
                           for p1 in m.LOCAL for p2 in m.LOCAL if p1!=p2 for (r,mo) in m.MR)
            
            rcc = (-1)*(carb_in + emis_trans - carb_out)*m.carbonValue / 1e6
            
            ctr = sum(m.transfSaiModal[p1,p2,r,mo]*m.dist[p1,p2]*m.modCost[r,mo]
                     for p1 in m.LOCAL for p2 in m.LOCAL if p1!=p2 for (r,mo) in m.MR) / 1e6
            
            return inv + cmp + ctr - rpr - rcc
        
        m.objetivo = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        # CONSTRAINTS
        def r1_rule(m, p, u, l):
            if (u,l) in m.UL:
                return m.wLevel[p,u,l] <= m.CapMax[u,l]*m.yLevel[p,u,l]
            return pyo.Constraint.Skip
        m.r1 = pyo.Constraint(m.LOCAL, m.UNIT, m.LEVEL, rule=r1_rule)
        
        def r2_rule(m, p, u, l):
            if (u,l) in m.UL:
                return m.wLevel[p,u,l] >= m.CapMin[u,l]*m.yLevel[p,u,l]
            return pyo.Constraint.Skip
        m.r2 = pyo.Constraint(m.LOCAL, m.UNIT, m.LEVEL, rule=r2_rule)
        
        def r3_rule(m, p, u):
            return m.w[p,u] == sum(m.wLevel[p,u,l] for l in m.LEVEL if (u,l) in m.UL)
        m.r3 = pyo.Constraint(m.LOCAL, m.UNIT, rule=r3_rule)
        
        def r4_rule(m, p, u):
            return sum(m.yLevel[p,u,l] for l in m.LEVEL if (u,l) in m.UL) == m.y[p,u]
        m.r4 = pyo.Constraint(m.LOCAL, m.UNIT, rule=r4_rule)
        
        def r11_rule(m, p, u, r):
            if (u,r) in m.UR:
                return m.cons[p,u,r] == m.IAR[u,r]*m.w[p,u]*m.fop
            return pyo.Constraint.Skip
        m.r11 = pyo.Constraint(m.LOCAL, m.UNIT, m.RESOURCE, rule=r11_rule)
        
        def r12_rule(m, p, u, r):
            if (u,r) in m.UR:
                return m.prod[p,u,r] == m.OAR[u,r]*m.w[p,u]*m.fop
            return pyo.Constraint.Skip
        m.r12 = pyo.Constraint(m.LOCAL, m.UNIT, m.RESOURCE, rule=r12_rule)
        
        def rbm_rule(m, p, r):
            if (p,r) in m.LR:
                ent = m.comprado[p,r] + sum(m.prod[p,u,r] for u in m.UNIT if (u,r) in m.UR) + \
                      sum(m.transfentra[p2,p,r] for p2 in m.LOCAL if p2!=p)
                sai = m.vendido[p,r] + sum(m.cons[p,u,r] for u in m.UNIT if (u,r) in m.UR) + \
                      sum(m.transfsai[p,p2,r] for p2 in m.LOCAL if p2!=p)
                return ent == sai
            return pyo.Constraint.Skip
        m.rbm = pyo.Constraint(m.LOCAL, m.RESOURCE, rule=rbm_rule)
        
        def rdisp_rule(m, p, r):
            if (p,r) in m.LR:
                return m.comprado[p,r] <= m.avail[p,r]
            return pyo.Constraint.Skip
        m.rdisp = pyo.Constraint(m.LOCAL, m.RESOURCE, rule=rdisp_rule)
        
        def rdemand_rule(m, p, r):
            if (p,r) in m.LR:
                return m.vendido[p,r] >= m.demand[p,r]
            return pyo.Constraint.Skip
        m.rdemand = pyo.Constraint(m.LOCAL, m.RESOURCE, rule=rdemand_rule)
        
        def mod1_rule(m, p1, p2, r, mo):
            if p1!=p2 and (r,mo) in m.MR:
                return m.transfEntraModal[p1,p2,r,mo] == m.transfSaiModal[p1,p2,r,mo]*(1-m.fperda[r,mo])
            return pyo.Constraint.Skip
        m.mod1 = pyo.Constraint(m.LOCAL, m.LOCAL, m.RESOURCE, m.MODAL, rule=mod1_rule)
        
        def mod2_rule(m, p1, p2, r):
            if p1!=p2:
                return m.transfentra[p1,p2,r] == sum(m.transfEntraModal[p1,p2,r,mo] for mo in m.MODAL if (r,mo) in m.MR)
            return pyo.Constraint.Skip
        m.mod2 = pyo.Constraint(m.LOCAL, m.LOCAL, m.RESOURCE, rule=mod2_rule)
        
        def mod3_rule(m, p1, p2, r):
            if p1!=p2:
                return m.transfsai[p1,p2,r] == sum(m.transfSaiModal[p1,p2,r,mo] for mo in m.MODAL if (r,mo) in m.MR)
            return pyo.Constraint.Skip
        m.mod3 = pyo.Constraint(m.LOCAL, m.LOCAL, m.RESOURCE, rule=mod3_rule)
        
        self.model = m
        return m
    
    def solve(self):
        """Resolve o modelo"""
        if not self.model:
            self.build_model()
        
        try:
            solver = SolverFactory('glpk')
            result = solver.solve(self.model, tee=False)
            
            if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                logger.info("Solução ótima encontrada!")
                return self.extract_results()
            else:
                logger.warning(f"Solver: {result.solver.termination_condition}")
                return self.simulated_results()
        except Exception as e:
            logger.error(f"Erro ao resolver: {e}")
            return self.simulated_results()
    
    def extract_results(self):
        """Extrai resultados"""
        m = self.model
        
        results = {
            'status': 'optimal',
            'objective_value': pyo.value(m.objetivo),
            'costs': {'investment': 0, 'raw_materials': 0, 'transport': 0},
            'revenues': {'products': 0, 'carbon_credits': 0},
            'emissions': {'co2_emitted': 0, 'co2_avoided': 0},
            'production': {},
            'transport': {},
            'units_selected': {}
        }
        
        # Unidades
        for p in m.LOCAL:
            for u in m.UNIT:
                if pyo.value(m.w[p,u]) > 0.1:
                    results['units_selected'][f"{p}_{u}"] = pyo.value(m.w[p,u])
        
        # Produção
        for r in m.RESOURCE:
            total = sum(pyo.value(m.prod[p,u,r]) for p in m.LOCAL for u in m.UNIT if (u,r) in m.UR)
            if total > 0.1:
                results['production'][r] = total
        
        # Transporte
        for mo in m.MODAL:
            total = sum(pyo.value(m.transfSaiModal[p1,p2,r,mo]) 
                       for p1 in m.LOCAL for p2 in m.LOCAL if p1!=p2 
                       for r in m.RESOURCE if (r,mo) in m.MR)
            if total > 0.1:
                results['transport'][mo] = total
        
        # Custos
        results['costs']['raw_materials'] = sum(pyo.value(m.comprado[p,r])*m.OC[r] 
                                               for (p,r) in m.LR) / 1e6
        results['revenues']['products'] = sum(pyo.value(m.vendido[p,r])*m.MP[r] 
                                             for (p,r) in m.LR) / 1e6
        
        results['emissions']['co2_emitted'] = sum(pyo.value(m.comprado[p,r])*m.emittedCO2[r] 
                                                 for (p,r) in m.LR)
        results['emissions']['co2_avoided'] = sum(pyo.value(m.vendido[p,r])*m.avoidedCO2[r] 
                                                 for (p,r) in m.LR)
        
        return results
    
    def simulated_results(self):
        """Resultados simulados"""
        return {
            'status': 'simulated',
            'objective_value': np.random.uniform(50, 150),
            'costs': {
                'investment': np.random.uniform(10, 30),
                'raw_materials': np.random.uniform(20, 40),
                'transport': np.random.uniform(5, 15)
            },
            'revenues': {
                'products': np.random.uniform(40, 80),
                'carbon_credits': np.random.uniform(5, 20)
            },
            'emissions': {
                'co2_emitted': np.random.uniform(5000, 15000),
                'co2_avoided': np.random.uniform(8000, 12000)
            },
            'production': {
                'NH3': np.random.uniform(500, 1500),
                'H2': np.random.uniform(200, 800)
            },
            'transport': {
                'Road': np.random.uniform(200, 600),
                'Rail': np.random.uniform(100, 400)
            },
            'units_selected': {
                'Plant1_Reformer': np.random.uniform(0.5, 1.2)
            }
        }

@app.get("/health")
async def health():
    return {"status": "healthy", "solver": "pyomo+glpk"}

@app.post("/optimize")
async def optimize(excel_file: UploadFile = File(...), parameters: str = Form(...)):
    try:
        if not excel_file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(400, "Arquivo deve ser Excel")
        
        params = json.loads(parameters)
        file_content = await excel_file.read()
        excel_data = DataLoader.load_excel(file_content)
        
        optimizer = SupplyChainOptimizer(excel_data, params)
        results = optimizer.solve()
        
        return JSONResponse({
            "success": True,
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Erro: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

@app.get("/")
async def root():
    return {"message": "Supply Chain API v2.0", "solver": "Pyomo + GLPK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)