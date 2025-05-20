import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PortfolioResultsAnalyzer:
    """
    Clase para analizar y visualizar los resultados del modelo de portafolio
    """
    
    def __init__(self, window_results, tickers):
        """
        Args:
            window_results: Lista de resultados por ventana del modelo
            tickers: Lista de nombres de los activos
        """
        self.window_results = window_results
        self.tickers = tickers
        self.df_results = self._create_results_dataframe()
    
    def _create_results_dataframe(self):
        """Convierte los resultados a DataFrame para facilitar el análisis"""
        data = []
        for res in self.window_results:
            row = {
                'window': res['window'],
                'test_start': res['test_start_date'],
                'test_end': res['test_end_date'],
                'total_return': res['metrics']['total_return'],
                'test_reward': res['metrics']['test_reward'],
                'test_er': res['metrics']['test_er'],
                'test_cvar': res['metrics']['test_cvar'],
                'test_diversity': res['metrics']['test_diversity']
            }
            
            # Añadir pesos por activo
            for i, ticker in enumerate(self.tickers):
                row[f'weight_{ticker}'] = res['weights'][i]
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_performance_overview(self, figsize=(15, 10)):
        """Gráfico general de rendimiento por ventana"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Resumen de Rendimiento del Modelo por Ventana', fontsize=16, fontweight='bold')
        
        # 1. Rendimientos totales por ventana
        axes[0, 0].bar(self.df_results['window'], self.df_results['total_return'] * 100)
        axes[0, 0].set_title('Rendimiento Total por Ventana (%)')
        axes[0, 0].set_xlabel('Ventana')
        axes[0, 0].set_ylabel('Rendimiento (%)')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 2. Reward de prueba por ventana
        axes[0, 1].plot(self.df_results['window'], self.df_results['test_reward'], marker='o')
        axes[0, 1].set_title('Reward de Prueba por Ventana')
        axes[0, 1].set_xlabel('Ventana')
        axes[0, 1].set_ylabel('Test Reward')
        
        # 3. Rendimiento esperado vs CVaR
        scatter = axes[1, 0].scatter(self.df_results['test_cvar'], self.df_results['test_er'], 
                                   c=self.df_results['window'], cmap='viridis', s=60)
        axes[1, 0].set_title('Rendimiento Esperado vs CVaR')
        axes[1, 0].set_xlabel('CVaR')
        axes[1, 0].set_ylabel('Rendimiento Esperado')
        plt.colorbar(scatter, ax=axes[1, 0], label='Ventana')
        
        # 4. Distribución de rendimientos
        axes[1, 1].hist(self.df_results['total_return'] * 100, bins=8, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribución de Rendimientos (%)')
        axes[1, 1].set_xlabel('Rendimiento (%)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def plot_weights_evolution(self, figsize=(14, 8)):
        """Evolución de los pesos del portafolio a lo largo de las ventanas"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle('Evolución de los Pesos del Portafolio', fontsize=16, fontweight='bold')
        
        # 1. Gráfico de líneas para ver evolución
        for ticker in self.tickers:
            weights = self.df_results[f'weight_{ticker}'] * 100
            ax1.plot(self.df_results['window'], weights, marker='o', label=ticker, linewidth=2)
        
        ax1.set_title('Evolución de Pesos por Ventana (%)')
        ax1.set_xlabel('Ventana')
        ax1.set_ylabel('Peso (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Mapa de calor de pesos
        weight_matrix = []
        for _, row in self.df_results.iterrows():
            weights = [row[f'weight_{ticker}'] * 100 for ticker in self.tickers]
            weight_matrix.append(weights)
        
        weight_matrix = np.array(weight_matrix).T
        im = ax2.imshow(weight_matrix, cmap='RdYlBu_r', aspect='auto')
        ax2.set_title('Mapa de Calor de Pesos (%)')
        ax2.set_xlabel('Ventana')
        ax2.set_ylabel('Activo')
        ax2.set_yticks(range(len(self.tickers)))
        ax2.set_yticklabels(self.tickers)
        
        # Añadir valores en el mapa de calor
        for i in range(len(self.tickers)):
            for j in range(len(self.df_results)):
                text = ax2.text(j, i, f'{weight_matrix[i, j]:.1f}%',
                              ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='Peso (%)')
        plt.tight_layout()
        plt.show()
    
    def plot_risk_return_analysis(self, figsize=(12, 8)):
        """Análisis de riesgo-rendimiento"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Análisis de Riesgo-Rendimiento', fontsize=16, fontweight='bold')
        
        # 1. Frontera eficiente simulada
        axes[0, 0].scatter(self.df_results['test_cvar'], self.df_results['test_er'] * 100, 
                          c=self.df_results['test_diversity'], cmap='plasma', s=80)
        axes[0, 0].set_title('Puntos Riesgo-Rendimiento por Diversificación')
        axes[0, 0].set_xlabel('CVaR')
        axes[0, 0].set_ylabel('Rendimiento Esperado (%)')
        
        # 2. Diversificación vs Rendimiento
        axes[0, 1].scatter(self.df_results['test_diversity'], self.df_results['total_return'] * 100, 
                          alpha=0.7, s=60)
        axes[0, 1].set_title('Diversificación vs Rendimiento Total')
        axes[0, 1].set_xlabel('Índice de Diversificación')
        axes[0, 1].set_ylabel('Rendimiento Total (%)')
        
        # 3. CVaR vs Rendimiento Total
        axes[1, 0].scatter(self.df_results['test_cvar'], self.df_results['total_return'] * 100, 
                          alpha=0.7, s=60, color='coral')
        axes[1, 0].set_title('CVaR vs Rendimiento Total')
        axes[1, 0].set_xlabel('CVaR')
        axes[1, 0].set_ylabel('Rendimiento Total (%)')
        
        # 4. Box plot de métricas
        metrics_data = [
            self.df_results['total_return'] * 100,
            self.df_results['test_er'] * 100,
            self.df_results['test_cvar'] * 100,
            self.df_results['test_diversity'] * 100
        ]
        axes[1, 1].boxplot(metrics_data, labels=['Return Total', 'E[r]', 'CVaR', 'Diversidad'])
        axes[1, 1].set_title('Distribución de Métricas (%)')
        axes[1, 1].set_ylabel('Valor (%)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_cumulative_performance(self, figsize=(12, 6)):
        """Rendimiento acumulativo del portafolio"""
        # Calcular rendimiento acumulativo
        returns = self.df_results['total_return'].values
        cumulative_return = np.cumprod(1 + returns) - 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Análisis de Rendimiento Acumulativo', fontsize=16, fontweight='bold')
        
        # 1. Rendimiento acumulativo
        ax1.plot(self.df_results['window'], cumulative_return * 100, marker='o', linewidth=2, markersize=6)
        ax1.set_title('Rendimiento Acumulativo (%)')
        ax1.set_xlabel('Ventana')
        ax1.set_ylabel('Rendimiento Acumulativo (%)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 2. Drawdown analysis
        running_max = np.maximum.accumulate(1 + cumulative_return)
        drawdown = ((1 + cumulative_return) / running_max - 1) * 100
        
        ax2.fill_between(self.df_results['window'], drawdown, 0, alpha=0.3, color='red')
        ax2.plot(self.df_results['window'], drawdown, color='darkred', linewidth=2)
        ax2.set_title('Drawdown del Portafolio (%)')
        ax2.set_xlabel('Ventana')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_statistics(self):
        """Imprime estadísticas resumen"""
        returns = self.df_results['total_return']
        
        print("=" * 60)
        print("ESTADÍSTICAS RESUMEN DEL MODELO")
        print("=" * 60)
        
        print(f"Número de ventanas evaluadas: {len(self.df_results)}")
        print(f"Rendimiento promedio: {returns.mean():.2%}")
        print(f"Desviación estándar: {returns.std():.2%}")
        print(f"Rendimiento mínimo: {returns.min():.2%}")
        print(f"Rendimiento máximo: {returns.max():.2%}")
        print(f"Ratio Sharpe (aproximado): {returns.mean() / returns.std():.3f}")
        
        positive_returns = (returns > 0).sum()
        negative_returns = (returns <= 0).sum()
        print(f"Ventanas positivas: {positive_returns} ({positive_returns/len(returns):.1%})")
        print(f"Ventanas negativas: {negative_returns} ({negative_returns/len(returns):.1%})")
        
        # Rendimiento acumulativo
        cumulative_return = np.prod(1 + returns) - 1
        print(f"Rendimiento acumulativo total: {cumulative_return:.2%}")
        
        # Métricas de riesgo promedio
        print(f"\nMétricas de riesgo promedio:")
        print(f"CVaR promedio: {self.df_results['test_cvar'].mean():.4f}")
        print(f"Diversificación promedio: {self.df_results['test_diversity'].mean():.4f}")
        print(f"Reward promedio: {self.df_results['test_reward'].mean():.4f}")
        
        # Análisis de pesos
        print(f"\nAnálisis de pesos promedio:")
        for ticker in self.tickers:
            avg_weight = self.df_results[f'weight_{ticker}'].mean()
            std_weight = self.df_results[f'weight_{ticker}'].std()
            print(f"{ticker}: {avg_weight:.2%} ± {std_weight:.2%}")

# Función principal para usar el analizador
def analyze_portfolio_results(window_results, tickers):
    """
    Función principal para analizar los resultados del modelo de portafolio
    
    Args:
        window_results: Lista de resultados por ventana (del modelo.py)
        tickers: Lista de nombres de los activos
    """
    
    # Verificar que hay resultados
    if not window_results:
        print("No hay resultados para analizar.")
        return
    
    # Crear el analizador
    analyzer = PortfolioResultsAnalyzer(window_results, tickers)
    
    # Mostrar estadísticas resumen
    analyzer.print_summary_statistics()
    
    # Generar todos los gráficos
    print("\nGenerando gráficos...")
    
    analyzer.plot_performance_overview()
    analyzer.plot_weights_evolution()
    analyzer.plot_risk_return_analysis()
    analyzer.plot_cumulative_performance()
    
    print("¡Análisis completo!")
    
    return analyzer

# Ejemplo de uso con datos simulados para prueba
if __name__ == "__main__":
    # Crear datos de ejemplo para mostrar cómo funciona
    import random
    from datetime import datetime, timedelta
    
    # Configuración de ejemplo
    tickers = ["TSLA", "ARKK", "AAPL", "PG", "JNJ", "ETH-USD", "BIL", "SHV", "GLD", "NOC"]
    n_windows = 10
    
    # Simular resultados de ejemplo
    window_results_example = []
    
    for i in range(n_windows):
        # Generar pesos aleatorios normalizados
        weights = np.random.random(len(tickers))
        weights = weights / weights.sum()
        
        # Generar métricas simuladas
        test_return = random.uniform(-0.05, 0.08)  # -5% a 8%
        test_er = random.uniform(-0.001, 0.003)
        test_cvar = random.uniform(0.01, 0.04)
        test_diversity = 1 - np.sum(weights**2)
        test_reward = test_er - 0.3 * test_cvar + 0.2 * test_diversity
        
        # Fechas simuladas
        start_date = datetime(2023, 1, 1) + timedelta(days=i*30)
        end_date = start_date + timedelta(days=10)
        
        result = {
            'window': i + 1,
            'train_end_date': start_date - timedelta(days=1),
            'test_start_date': start_date,
            'test_end_date': end_date,
            'weights': weights,
            'metrics': {
                'total_return': test_return,
                'test_reward': test_reward,
                'test_er': test_er,
                'test_cvar': test_cvar,
                'test_diversity': test_diversity
            }
        }
        window_results_example.append(result)
    
    # Ejecutar el análisis con datos de ejemplo
    print("Ejecutando análisis con datos de ejemplo...")
    analyzer = analyze_portfolio_results(window_results_example, tickers)
    
    print("\n" + "="*60)
    print("INSTRUCCIONES DE USO:")
    print("="*60)
    print("Para usar este código con los resultados reales de model.py:")
    print("1. Ejecuta model.py y guarda la variable 'window_results'")
    print("2. Importa este módulo: from portfolio_visualization import analyze_portfolio_results")
    print("3. Ejecuta: analyze_portfolio_results(window_results, tickers)")
    print("="*60)