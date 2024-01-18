from logging import Logger, config, getLogger

from .config import APPLICATION_NAME, CONFIGS

_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "'%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "default": {
            "level": CONFIGS.log_level,
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": CONFIGS.log_level,
        },
        APPLICATION_NAME: {
            "handlers": ["default"],
            "level": CONFIGS.log_level,
        },
    },
}


config.dictConfig(_LOG_CONFIG)


logger = getLogger(APPLICATION_NAME)


def get_logger(name: str) -> Logger:
    return getLogger(name=name)
