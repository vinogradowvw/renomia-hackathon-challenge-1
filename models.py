from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker
from sqlalchemy import Text, create_engine, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from datetime import datetime
from contextvars import ContextVar
import logging
from contextlib import contextmanager

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

session_context_var: ContextVar[Session | None] = ContextVar("db_session", default=None)

_engine = None
_Session = None

class Base(DeclarativeBase):
    pass

class Cache(Base):
    __tablename__ = "cache"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

def get_engine(database_url: str = None):
    global _engine
    if _engine is None:
        url = database_url or config.DATABASE_URL
        _engine = create_engine(url, echo=False)
    return _engine

def get_session():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _Session()

@contextmanager
def transaction():
    session = session_context_var.get()
    created_here = session is None
    save_point = None
    token = None

    if created_here:
        session = get_session()
        token = session_context_var.set(session)

    is_nested = session.in_transaction()

    try:
        if is_nested:
            save_point = session.begin_nested()
        else:
            session.begin()

        yield session

        if is_nested:
            save_point.commit()
        else:
            session.commit()

    except Exception as e:
        logger.exception(f"Exception occurred during transaction, rolling back: {str(e)}")
        
        try:
            if is_nested and save_point is not None:
                save_point.rollback()
            else:
                session.rollback()
        except Exception:
            logger.exception("Rollback failed")
        raise

    finally:
        if created_here:
            session.close()
            session_context_var.reset(token)