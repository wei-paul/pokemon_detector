import { Sequelize, DataTypes } from 'sequelize';

const sequelize = new Sequelize(process.env.POSTGRES_DB, process.env.POSTGRES_USER, process.env.POSTGRES_PASSWORD, {
    host: 'db',
    dialect: 'postgres'
});

const History = sequelize.define('sys_history', {
    originalImage: DataTypes.STRING,
    detectedPokemon: DataTypes.STRING,
    confidence: DataTypes.FLOAT,
    timestamp: DataTypes.DATE
});

await sequelize.sync();

export { sequelize, History };