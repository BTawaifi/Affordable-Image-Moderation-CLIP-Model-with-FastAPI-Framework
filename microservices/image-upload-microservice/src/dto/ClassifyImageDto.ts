import { IsOptional, IsString } from 'class-validator';
import { Transform, Type } from 'class-transformer';

export class ClassifyImageDto {
  @Transform(({ value }) => value.split(',').map((item: string) => item.trim()))
  @Type(() => String)
  @IsString({ each: true })
  classes: string;

  @IsOptional()
  @Transform(({ value }) => value.split(',').map((item: string) => item.trim()))
  @Type(() => String)
  @IsString({ each: true })
  flaggingThreshold: string;
}
