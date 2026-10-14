import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ImageRetrievalController } from './image-retrieval.controller';
import { ImageRetrievalService } from './image-retrieval.service';
import { ClassificationResult } from './entities/classification-results.entity';
import { LabelProbability } from './entities/label-probabilities.entity';

@Module({
  imports: [TypeOrmModule.forFeature([ClassificationResult, LabelProbability])],
  controllers: [ImageRetrievalController],
  providers: [ImageRetrievalService],
})
export class ImageRetrievalModule { }
