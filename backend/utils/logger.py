import sys
import logging
import colorlog

def configure_logger() -> logging.Logger:
    """
    Configure la journalisation application-wide avec des messages colorés pour AI-MAP.
    
    Returns:
        logging.Logger: L'instance du logger configurée.
    """
    # Créer un logger
    config_logger = logging.getLogger(__name__)
    config_logger.setLevel(logging.DEBUG)  # Définir à DEBUG pour capturer tous les niveaux
    
    # Empêcher les handlers dupliqués
    if config_logger.handlers:
        return config_logger

    # Créer un formateur coloré
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',       # Messages de debug en cyan
            'INFO': 'green',       # Messages d'info en vert
            'WARNING': 'yellow',   # Messages d'avertissement en jaune
            'ERROR': 'red',        # Messages d'erreur en rouge
            'CRITICAL': 'red',     # Messages critiques en rouge
        }
    )

    # Créer un handler console et définir le formateur
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    
    # Ajouter le handler au logger
    config_logger.addHandler(ch)
    
    return config_logger

# Initialiser le logger
logger = configure_logger()

# Exemple d'utilisation
if __name__ == '__main__':
    logger.debug('Ceci est un message de debug')      # Cyan
    logger.info('Ceci est un message d\'information')       # Vert
    logger.warning('Ceci est un message d\'avertissement')  # Jaune
    logger.error('Ceci est un message d\'erreur')     # Rouge
    logger.critical('Ceci est un message critique')  # Rouge
