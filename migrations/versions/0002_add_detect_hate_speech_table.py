"""add detect hate speech table

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-21 21:54:04.532462

"""
from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "detect_hate_speech_result",
        sa.Column("is_hate_speech", sa.Boolean(), nullable=False),
        sa.Column("target_of_hate", sa.ARRAY(sa.String()), nullable=True),
        sa.Column("reasoning", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("text", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("uuid", sqlmodel.sql.sqltypes.GUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("llm_run_id", sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["llm_run_id"],
            ["llm_run.uuid"],
        ),
        sa.PrimaryKeyConstraint("uuid"),
    )
    op.create_index(op.f("ix_detect_hate_speech_result_uuid"), "detect_hate_speech_result", ["uuid"], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_detect_hate_speech_result_uuid"), table_name="detect_hate_speech_result")
    op.drop_table("detect_hate_speech_result")
    # ### end Alembic commands ###
