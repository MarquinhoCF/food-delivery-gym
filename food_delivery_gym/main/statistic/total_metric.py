from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class TotalMetric(Metric):

    def __init__(self, environment: FoodDeliverySimpyEnv, table=False):
        super().__init__(environment)
        self.table = table

    def view(self, ax) -> None:
        print("TOTAL ESTABLISHMENTS", len(self.environment.state.establishments))
        print("TOTAL CUSTOMERS", len(self.environment.state.customers))
        print("TOTAL DRIVERS", len(self.environment.state.drivers))
        print("TOTAL ORDERS", len(self.environment.state.orders))

        labels = ['Establishments', 'Customers', 'Drivers', 'Orders']
        values = [len(self.environment.state.establishments), len(self.environment.state.customers),
                  len(self.environment.state.drivers), len(self.environment.state.orders)]

        ax.set_title('Total generated data')

        if self.table:
            ax.axis('tight')
            ax.axis('off')
            table_data = [[label, value] for label, value in zip(labels, values)]
            table = ax.table(cellText=table_data, colLabels=['Entities', 'Total'], cellLoc='center', loc='center')
            # table.auto_set_font_size(False)
            # table.set_fontsize(12)
            # table.scale(1.2, 1.2)
        else:
            ax.barh(labels, values, align='center')
            ax.set_xlabel('Total')
            ax.set_ylabel('Entities')



