
# ------------------------------------------------------------------------
# Energy management environment for reinforcement learning agents developed by
# Hou Shengren, TU Delft, h.shengren@tudelft.nl
import random
import numpy as np
import pandas as pd 
import gym
from gym import spaces 

from Parameters import battery_parameters,dg_parameters

class Constant:
	MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	MAX_STEP_HOURS = 24 * 30
class DataManager():
    def __init__(self) -> None:
        self.PV_Generation=[]
        self.Prices=[]
        self.Electricity_Consumption=[]
    def __len__(self):
        return len(self.Prices)
    def add_pv_element(self,element):self.PV_Generation.append(element)
    def add_price_element(self,element):self.Prices.append(element)
    def add_electricity_element(self,element):self.Electricity_Consumption.append(element)

    # get current time data based on given month day, and day_time
    def get_pv_data(self,month,day,day_time):return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_price_data(self,month,day,day_time):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_electricity_cons_data(self,month,day,day_time):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    # get series data for one episode
    def get_series_pv_data(self,month,day): return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_price_data(self,month,day):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_electricity_cons_data(self,month,day):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_tuple_by_index(self,idx):
        total_hours=len(self.PV_Generation)
        real_idx=idx%total_hours
        return (self.PV_Generation[real_idx],
                self.Prices[real_idx],
                self.Electricity_Consumption[real_idx])

class DG():
    '''simulate a simple diesel generator here'''
    def __init__(self,parameters):
        self.name=parameters.keys()
        self.a_factor=parameters['a']
        self.b_factor=parameters['b']
        self.c_factor=parameters['c']
        self.power_output_max=parameters['power_output_max']
        self.power_output_min=parameters['power_output_min']
        self.ramping_up=parameters['ramping_up']
        self.ramping_down=parameters['ramping_down']
        self.last_step_output=None 
    def step(self,action_gen):
        output_change=action_gen*self.ramping_up# constrain the output_change with ramping up boundary
        output=self.current_output+output_change
        if output>0:
            output=max(self.power_output_min,min(self.power_output_max,output))# meet the constrain 
        else:
            output=0
        self.current_output=output
    def _get_cost(self,output):
        if output<=0:
            cost=0
        else:
            cost=(self.a_factor*pow(output,2)+self.b_factor*output+self.c_factor)
        return cost 
    def reset(self):
        self.current_output=0

class Battery():
    '''simulate a simple battery here'''
    def __init__(self,parameters):
        self.capacity=parameters['capacity']
        self.max_soc=parameters['max_soc']
        self.initial_capacity=parameters['initial_capacity']
        self.min_soc=parameters['min_soc']# 0.2
        self.degradation=parameters['degradation']# degradation cost 1.2
        self.max_charge=parameters['max_charge']# nax charge ability
        self.max_discharge=parameters['max_discharge']
        self.efficiency=parameters['efficiency']
    def step(self,action_battery):
        energy=action_battery*self.max_charge
        updated_capacity=max(self.min_soc,min(self.max_soc,(self.current_capacity*self.capacity+energy)/self.capacity))
        self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge, positive, if discharge, negative
        self.current_capacity=updated_capacity# update capacity to current codition
    def _get_cost(self,energy):# calculate the cost depends on the energy change
        cost=energy**2*self.degradation
        return cost  
    def SOC(self):
        return self.current_capacity
    def reset(self):
        self.current_capacity=np.random.uniform(0.2,0.8)
class Grid():
    def __init__(self):
        
        self.on=True
        if self.on:
            self.exchange_ability=100
        else:
            self.exchange_ability=0
    def _get_cost(self,current_price,energy_exchange):
        return current_price*energy_exchange
    def retrive_past_price(self):
        result=[]
        if self.day<1:
            past_price=self.past_price#
        else:
            past_price=self.price[24*(self.day-1):24*self.day]
            # print(past_price)
        for item in past_price[(self.time-24)::]:
            result.append(item)
        for item in self.price[24*self.day:(24*self.day+self.time)]:
            result.append(item)
        return result 
class ESSEnv(gym.Env):
    def __init__(self,**kwargs):
        super().__init__()
        self.data_manager=DataManager()
        self._load_year_data()

        self.episode_length=kwargs.get('episode_length',24)
        self.forecast_horizon=kwargs.get('forecast_horizon',4)
        self.use_time_features=kwargs.get('use_time_features',True)
        self.normalize_observation=kwargs.get('normalize_observation',True)

        self.month=None
        self.day=None
        self.TRAIN=True
        self.current_time=0
        self._pv_series=None
        self._price_series=None
        self._load_series=None
        self._day_start_hour=0
        self._year_hours=len(self.data_manager) or 1

        self.battery_parameters=kwargs.get('battery_parameters',battery_parameters)
        self.dg_parameters=kwargs.get('dg_parameters',dg_parameters)
        self.penalty_coefficient=kwargs.get('penalty_coefficient',50)
        self.sell_coefficient=kwargs.get('sell_coefficient',0.5)

        self.grid=Grid()
        self.battery=Battery(self.battery_parameters)
        self.dg1=DG(self.dg_parameters['gen_1'])
        self.dg2=DG(self.dg_parameters['gen_2'])
        self.dg3=DG(self.dg_parameters['gen_3'])

        self.action_space=spaces.Box(low=-1,high=1,shape=(4,),dtype=np.float32)

        self.price_scale=max(self.data_manager.Prices) if self.data_manager.Prices else 1.0
        self.load_scale=max(self.data_manager.Electricity_Consumption) if self.data_manager.Electricity_Consumption else 1.0
        self.pv_scale=max(self.data_manager.PV_Generation) if self.data_manager.PV_Generation else 1.0
        self.netload_scale=max(self.load_scale,1.0)
        self.dg_max_outputs=np.array([self.dg1.power_output_max,self.dg2.power_output_max,self.dg3.power_output_max],dtype=np.float32)

        self.state_dim=self._calc_state_dim()
        if self.normalize_observation:
            self.state_space=spaces.Box(low=-1.0,high=1.0,shape=(self.state_dim,),dtype=np.float32)
        else:
            self.state_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.state_dim,),dtype=np.float32)

    def _calc_state_dim(self):
        time_dim=2 if self.use_time_features else 1
        base_dim=time_dim+1+1+1+3
        forecast_dim=3*self.forecast_horizon
        return base_dim+forecast_dim

    def _month_day_to_hour_index(self,month,day):
        return (sum(Constant.MONTHS_LEN[:month-1])+day-1)*24

    def _hour_index(self,offset=0):
        return (self._day_start_hour+self.current_time+offset)%self._year_hours

    def reset(self,):
        self.month=np.random.randint(1,13)
        if self.TRAIN:
            self.day=np.random.randint(1,20)
        else:
            self.day=np.random.randint(20,Constant.MONTHS_LEN[self.month]-1)
        self.current_time=0
        self._day_start_hour=self._month_day_to_hour_index(self.month,self.day)
        self._pv_series=np.asarray(self.data_manager.get_series_pv_data(self.month,self.day),dtype=np.float32)
        self._price_series=np.asarray(self.data_manager.get_series_price_data(self.month,self.day),dtype=np.float32)
        self._load_series=np.asarray(self.data_manager.get_series_electricity_cons_data(self.month,self.day),dtype=np.float32)
        self.battery.reset()
        self.dg1.reset()
        self.dg2.reset()
        self.dg3.reset()
        return self._build_state()

    def _get_current_measurements(self):
        pv_generation=self._pv_series[self.current_time]
        price=self._price_series[self.current_time]
        electricity_demand=self._load_series[self.current_time]
        net_load=electricity_demand-pv_generation
        return pv_generation,price,electricity_demand,net_load

    def _normalize(self,value,scale):
        if not self.normalize_observation or scale==0:
            return np.float32(value)
        return np.float32(np.clip(value/scale,-1.0,1.0))

    def _get_forecast_features(self):
        if self.forecast_horizon<=0:
            return []
        features=[]
        for offset in range(1,self.forecast_horizon+1):
            pv,price,load=self.data_manager.get_tuple_by_index(self._hour_index(offset))
            net_load=load-pv
            features.extend([self._normalize(price,self.price_scale),
                             self._normalize(net_load,self.netload_scale),
                             self._normalize(pv,self.pv_scale)])
        return features

    def _build_state(self,measurement=None):
        measurement=measurement or self._get_current_measurements()
        pv_generation,price,_,net_load=measurement
        soc=self.battery.SOC()
        dg_outputs=np.array((self.dg1.current_output,self.dg2.current_output,self.dg3.current_output),dtype=np.float32)
        dg_norm=dg_outputs/np.maximum(self.dg_max_outputs,1e-6)
        if self.normalize_observation:
            dg_norm=np.clip(dg_norm,0.0,1.0)
        time_fraction=self.current_time/self.episode_length
        state_components=[]
        if self.use_time_features:
            angle=2*np.pi*time_fraction
            state_components.extend([np.sin(angle),np.cos(angle)])
        else:
            state_components.append(np.float32(time_fraction))
        state_components.append(self._normalize(price,self.price_scale))
        state_components.append(np.float32(soc))
        state_components.append(self._normalize(net_load,self.netload_scale))
        state_components.extend(dg_norm.tolist())
        state_components.extend(self._get_forecast_features())
        return np.array(state_components,dtype=np.float32)

    def step(self,action):
        measurement=self._get_current_measurements()
        current_state=self._build_state(measurement)
        self.battery.step(action[0])
        self.dg1.step(action[1])
        self.dg2.step(action[2])
        self.dg3.step(action[3])
        current_output=np.array((self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,-self.battery.energy_change))
        self.current_output=current_output
        actual_production=sum(current_output)
        pv_generation,price,electricity_demand,netload=measurement

        unbalance=actual_production-netload

        reward=0
        excess_penalty=0
        deficient_penalty=0
        sell_benefit=0
        buy_cost=0
        self.excess=0
        self.shedding=0
        if unbalance>=0:
            if unbalance<=self.grid.exchange_ability:
                sell_benefit=self.grid._get_cost(price,unbalance)*self.sell_coefficient
            else:
                sell_benefit=self.grid._get_cost(price,self.grid.exchange_ability)*self.sell_coefficient
                self.excess=unbalance-self.grid.exchange_ability
                excess_penalty=self.excess*self.penalty_coefficient
        else:
            if abs(unbalance)<=self.grid.exchange_ability:
                buy_cost=self.grid._get_cost(price,abs(unbalance))
            else:
                buy_cost=self.grid._get_cost(price,self.grid.exchange_ability)
                self.shedding=abs(unbalance)-self.grid.exchange_ability
                deficient_penalty=self.shedding*self.penalty_coefficient
        battery_cost=self.battery._get_cost(self.battery.energy_change)
        dg1_cost=self.dg1._get_cost(self.dg1.current_output)
        dg2_cost=self.dg2._get_cost(self.dg2.current_output)
        dg3_cost=self.dg3._get_cost(self.dg3.current_output)

        reward-=(battery_cost+dg1_cost+dg2_cost+dg3_cost+excess_penalty+deficient_penalty-sell_benefit+buy_cost)/1e3
        self.operation_cost=battery_cost+dg1_cost+dg2_cost+dg3_cost+buy_cost-sell_benefit+excess_penalty+deficient_penalty
        self.unbalance=unbalance
        self.real_unbalance=self.shedding+self.excess
        final_step_outputs=[self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,self.battery.current_capacity]

        info={
            'state':current_state,
            'time_step':self.current_time,
            'month':self.month,
            'day':self.day,
            'price':price,
            'netload':netload,
            'pv_generation':pv_generation,
            'electricity_demand':electricity_demand,
            'soc':self.battery.SOC(),
            'battery_energy_change':self.battery.energy_change,
            'dg_outputs':current_output[:3].copy(),
            'grid_exchange':unbalance,
            'operation_cost':self.operation_cost,
            'excess_load':self.excess,
            'shed_load':self.shedding
        }

        self.current_time+=1
        finish=(self.current_time==self.episode_length)
        if finish:
            self.final_step_outputs=final_step_outputs
            next_obs=self.reset()
        else:
            next_obs=self._build_state()
        return next_obs,float(reward),finish,info

    def render(self, action, reward, done, info=None):
        info=info or {}
        print('day={},hour={:2d}, action={}, reward={:.4f}, terminal={}, netload={:.2f}, unbalance={:.2f}'.format(
            self.day,self.current_time,action,reward,done,info.get('netload','n/a'),info.get('grid_exchange','n/a')))
    def _load_year_data(self):
        pv_df=pd.read_csv('data/PV.csv',sep=';')
        #hourly price data for a year 
        price_df=pd.read_csv('data/Prices.csv',sep=';')
        # mins electricity consumption data for a year 
        electricity_df=pd.read_csv('data/H4.csv',sep=';')
        pv_data=pv_df['P_PV_'].apply(lambda x: x.replace(',','.')).to_numpy(dtype=float)
        price=price_df['Price'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        electricity=electricity_df['Power'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        # netload=electricity-pv_data
        '''we carefully redesign the magnitude for price and amount of generation as well as demand'''
        for element in pv_data:
            self.data_manager.add_pv_element(element*200)
        for element in price:
            element/=10
            if element<=0.5:
                element=0.5
            self.data_manager.add_price_element(element)
        for i in range(0,electricity.shape[0],60):
            element=electricity[i:i+60]
            self.data_manager.add_electricity_element(sum(element)*300)
if __name__ == '__main__':
    env=ESSEnv()
    env.TRAIN=False
    rewards=[]

    current_obs=env.reset()
    tem_action=[0.1,0.1,0.1,0.1]
    for _ in range (144):
        print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time}')
        next_obs,reward,finish,info=env.step(tem_action)
        env.render(tem_action,reward,finish,info)
        current_obs=next_obs
        rewards.append(reward)
